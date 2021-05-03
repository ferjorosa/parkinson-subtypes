package eu.amidst.extension.learn.structure.operator.hc.tree;

import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.core.variables.stateSpaceTypes.FiniteStateSpace;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.VBEM_Global;
import eu.amidst.extension.learn.parameter.VBEM_Local;
import eu.amidst.extension.learn.structure.Result;
import eu.amidst.extension.learn.structure.typelocalvbem.TypeLocalVBEM;
import eu.amidst.extension.util.GraphUtilsAmidst;

import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;

/**
 * MyNote: Como se que un "variable.setNumberOfStates()" no cambia el plateau (aunque si afecta runInference), me aprovecho
 * para hacer el metodo algo mas eficiente. Sin embargo, no es lo correcto porque cuando llamo a vmp.runInference si no
 * coincide la cardinalidad de la variable con la cardinalidad en el platau, salta un IndexOutOfBoundsException. Es otra
 * razon mas por la que habria que relacionar ambos aspectos de forma similar a Votlric.
 *
 * Al terminar el aprendizaje y seleccionar el modelo mas adecuado, la cardinalidad de todas las variables ha sido restaurada,
 * por ello, previo a llamar VBEM_Global hay que volver a decrementarla porque sino saltará la excepcion.
 */
// TODO: El tema de la inicializacion en VBEM_Global no ha quedado resuelto. Deberia tomarse en cuenta el punto generado como una posibilidad mas
public class BltmHcDecreaseCard implements BltmHcOperator {

    private int minCardinality;

    private VBEMConfig localVBEMConfig;

    private VBEMConfig globalVBEMConfig;

    private TypeLocalVBEM typeLocalVBEM;

    public BltmHcDecreaseCard(int minCardinality, TypeLocalVBEM typeLocalVBEM) {
        this.minCardinality = minCardinality;
    }

    public BltmHcDecreaseCard(int minCardinality,
                              VBEMConfig localVBEMConfig,
                              VBEMConfig globalVBEMConfig,
                              TypeLocalVBEM typeLocalVBEM) {
        this.minCardinality = minCardinality;
        this.localVBEMConfig = localVBEMConfig;
        this.globalVBEMConfig = globalVBEMConfig;
        this.typeLocalVBEM = typeLocalVBEM;
    }

    @Override
    public Result apply(PlateuStructure plateuStructure, DAG dag, boolean doGlobalVBEM) {

        List<String> discreteLatentVars = dag.getVariables().getListOfVariables()
                .stream()
                .filter(x-> x.isDiscrete() && !x.isObservable())
                .map(x->x.getName())
                .collect(Collectors.toList());

        return apply(plateuStructure, dag, discreteLatentVars, doGlobalVBEM);
    }


    public Result apply(PlateuStructure plateuStructure, DAG dag, List<String> whiteList, boolean doGlobalVBEM) {

        PlateuStructure bestModel = null; // TODO: Cambio del 30-08, no deberia dar problema
        double bestModelScore = -Double.MAX_VALUE;
        Variable bestVariable = null;

        Variables copyVariables = dag.getVariables().deepCopy();
        DAG copyDAG = dag.deepCopy(copyVariables);

        /* Iteramos por el conjunto de variables latentes */
        for(Variable variable: copyVariables){

            if(!variable.isObservable()
                    && variable.isDiscrete()
                    && whiteList.contains(variable.getName())
                    && variable.getNumberOfStates() > this.minCardinality) {

                /* Decrementamos la cardinalidad de la variable */
                int newCardinality = variable.getNumberOfStates() - 1;
                variable.setNumberOfStates(newCardinality);
                variable.setStateSpaceType(new FiniteStateSpace(newCardinality));

                /* Creamos un nuevo Plateau para el aprendizaje donde omitimos copiar la variable en cuestion y sus hijos */
                HashSet<Variable> omittedVariables = new HashSet<>();
                omittedVariables.add(variable);
                omittedVariables.addAll(GraphUtilsAmidst.getChildren(variable, copyDAG));
                PlateuStructure copyPlateauStructure = plateuStructure.deepCopy(copyDAG, omittedVariables);

                /* Aprendemos el modelo de forma local */
                VBEM_Local localVBEM = new VBEM_Local(this.localVBEMConfig);
                localVBEM.learnModel(copyPlateauStructure, copyDAG, typeLocalVBEM.variablesToUpdate(variable, copyDAG));

                /* Comparamos el modelo generado con el mejor modelo actual */
                if(localVBEM.getPlateuStructure().getLogProbabilityOfEvidence() > bestModelScore) {
                    bestModel = localVBEM.getPlateuStructure();
                    bestModelScore = localVBEM.getPlateuStructure().getLogProbabilityOfEvidence();
                    bestVariable = variable;
                }

                /* Incrementamos la cardinalidad de la variable para poder resetear el proceso */
                variable.setNumberOfStates(newCardinality + 1);
                variable.setStateSpaceType(new FiniteStateSpace(newCardinality + 1));
            }
        }

        /* Si el operador produjo un mejor modelo, lo aprendemos con VBEM_HC de forma "global" */
        // TODO: Escoger el tipo de inicializacion a utilizar aqui, podria ser incluso NONE
        if(bestModelScore > -Double.MAX_VALUE) {

            // Decrementamos la cardinalidad de la mejor variable para que no salte una IndexOutOfBoundsException en VMP.runInference
            bestVariable.setNumberOfStates(bestVariable.getNumberOfStates() - 1);
            bestVariable.setStateSpaceType(new FiniteStateSpace(bestVariable.getNumberOfStates()));

            if(doGlobalVBEM) {
                VBEM_Global globalVBEM = new VBEM_Global(this.globalVBEMConfig);
                globalVBEM.learnModel(bestModel, copyDAG);

                bestModel = globalVBEM.getPlateuStructure();
                bestModelScore = globalVBEM.getPlateuStructure().getLogProbabilityOfEvidence();
            }
        }

        /* Devolvemos el resultado */
        return new Result(bestModel, bestModelScore, copyDAG, "DecreaseCard");
    }
}
