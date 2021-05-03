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

public class BltmHcIncreaseCard implements BltmHcOperator {

    private int maxCardinality;

    private VBEMConfig localVBEMConfig;

    private VBEMConfig globalVBEMConfig;

    private TypeLocalVBEM typeLocalVBEM;

    public BltmHcIncreaseCard(int maxCardinality, TypeLocalVBEM typeLocalVBEM) {
        this(maxCardinality,
                new VBEMConfig(),
                new VBEMConfig(),
                typeLocalVBEM);
    }

    public BltmHcIncreaseCard(int maxCardinality,
                              VBEMConfig localVBEMConfig,
                              VBEMConfig globalVBEMConfig,
                              TypeLocalVBEM typeLocalVBEM) {
        this.maxCardinality = maxCardinality;
        this.localVBEMConfig = localVBEMConfig;
        this.globalVBEMConfig = globalVBEMConfig;
        this.typeLocalVBEM = typeLocalVBEM;
    }

    /** ALGORITMO:
       La idea principal es la de aprender de forma local el mejor modelo generado por este operador.
       Una vez se ha escogido el mejor modelo local, lo podemos afinar con aprendizaje global

            Copias el objeto Variables
            Copias el objeto DAG

            Incrementas el numero de estados de la variable seleccionada
            Creas un nuevo plateau y copias aquellos aspectos que no han sido modificados por el incremento de card
            Aprendes y guardas el score
            Decrementas la cardinalidad de la variable seleccionada
            Pasas a la siguiente variable y vuelves a empezar

       RETURN:
       Para poder seguir trabajando con el modelo aprendido, se va a devolver la copia del Plateau
       modificado tras hacer un aprendizaje completo con VBEM.

       Por esto, creo que el mejor objeto de retorno deberia ser Result.

     MyNote: Al igual que en BltmHcDecreaseCard, me aprovecho de la separacion de Variables con Plateau para acelerar
     el proceso de copia del Plateau, pero en VBEM_Global tengo que restaurar la cardinalidad o lanza IndexOutOfBounds
     */
    @Override
    public Result apply(PlateuStructure plateuStructure, DAG dag, boolean globalVBEM) {

        List<String> discreteLatentVars = dag.getVariables().getListOfVariables()
                .stream()
                .filter(x-> x.isDiscrete() && !x.isObservable())
                .map(x->x.getName())
                .collect(Collectors.toList());

        return apply(plateuStructure, dag, discreteLatentVars, globalVBEM);
    }

    public Result apply(PlateuStructure plateuStructure, DAG dag, List<String> whiteList,  boolean globalVBEM) {

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
                    && variable.getNumberOfStates() < this.maxCardinality) {

                /* Incrementamos la cardinalidad de la variable */
                int newCardinality = variable.getNumberOfStates() + 1;
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

                /* Decrementamos la cardinalidad de la variable para poder resetear el proceso */
                variable.setNumberOfStates(newCardinality - 1);
                variable.setStateSpaceType(new FiniteStateSpace(newCardinality - 1));
            }
        }

        /* Si el operador produjo un mejor modelo, lo aprendemos con VBEM_HC de forma "global" */
        // TODO: Escoger el tipo de inicializacion a utilizar aqui, podria ser incluso NONE
        if(bestModelScore > -Double.MAX_VALUE) {

            // Incrementamos la cardinalidad de la mejor variable para que no salte una IndexOutOfBoundsException en VMP.runInference
            bestVariable.setNumberOfStates(bestVariable.getNumberOfStates() + 1);
            bestVariable.setStateSpaceType(new FiniteStateSpace(bestVariable.getNumberOfStates()));

            if(globalVBEM) {
                VBEM_Global vbem_hc = new VBEM_Global(this.globalVBEMConfig);
                vbem_hc.learnModel(bestModel, copyDAG);

                bestModel = vbem_hc.getPlateuStructure();
                bestModelScore = vbem_hc.getPlateuStructure().getLogProbabilityOfEvidence();
            }
        }

        /* Devolvemos el resultado */
        return new Result(bestModel, bestModelScore, copyDAG, "IncreaseCard");
    }
}
