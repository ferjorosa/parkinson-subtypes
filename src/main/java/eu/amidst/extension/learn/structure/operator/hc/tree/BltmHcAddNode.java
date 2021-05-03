package eu.amidst.extension.learn.structure.operator.hc.tree;

import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.VBEM_Global;
import eu.amidst.extension.learn.parameter.VBEM_Local;
import eu.amidst.extension.learn.structure.Result;
import eu.amidst.extension.learn.structure.typelocalvbem.TypeLocalVBEM;
import eu.amidst.extension.util.GraphUtilsAmidst;
import eu.amidst.extension.util.tuple.Tuple2;
import org.apache.commons.math3.util.Combinations;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;

/**
 * MyNote: Existen diferentes posibilidades para el aprendizaje de los parametros una vez se ha añadido un nuevo nodo
 * latente:
 *
 *      - Solo la nueva LV
 *      - La nueva LV, su padre y los hijos de su padre
 *      - El markov blanket de la nueva LV (incluyendola a ella)
 */
public class BltmHcAddNode implements BltmHcOperator {

    private int maxNumberOfLatentNodes;

    private VBEMConfig localVBEMConfig;

    private VBEMConfig globalVBEMConfig;

    private TypeLocalVBEM typeLocalVBEM;

    private int latentVarNameCounter = 0;

    public BltmHcAddNode(int maxNumberOfLatentNodes,
                         TypeLocalVBEM typeLocalVBEM) {
        this(maxNumberOfLatentNodes,
                new VBEMConfig(),
                new VBEMConfig(),
                typeLocalVBEM);
    }

    public BltmHcAddNode(int maxNumberOfLatentNodes,
                         VBEMConfig localVBEMConfig,
                         VBEMConfig globalVBEMConfig,
                         TypeLocalVBEM typeLocalVBEM) {
        this.maxNumberOfLatentNodes = maxNumberOfLatentNodes;
        this.localVBEMConfig = localVBEMConfig;
        this.globalVBEMConfig = globalVBEMConfig;
        this.typeLocalVBEM = typeLocalVBEM;
    }

    @Override
    public Result apply(PlateuStructure plateuStructure, DAG dag, boolean doGlobalVBEM) {

        PlateuStructure bestModel = plateuStructure;
        double bestModelScore = -Double.MAX_VALUE;
        Tuple2<Variable, Variable> bestPair = null;
        Variable bestPairParent = null;
        String newLatentVarName = "";

        /* En caso de que el numero permitido de variables latentes sea ya maximo, devolvemos el modelo actual como resultado */
        long numberOfLatentNodes = dag.getVariables().getListOfVariables().stream().filter(x->x.getAttribute() == null).count();
        if(numberOfLatentNodes >= maxNumberOfLatentNodes)
            return new Result(bestModel, bestModelScore, dag, "AddNode");

        Variables copyVariables = dag.getVariables().deepCopy();
        DAG copyDAG = dag.deepCopy(copyVariables);

        /* Iteramos por el conjunto de variables latentes */
        // No podemos iterar por copyVariables porque sino saltaria una excepcion al modificar la coleccion que estamos iterando
        for(Variable variable: dag.getVariables()){

            if(variable.getAttribute() == null) {

                List<Variable> observedChildren = GraphUtilsAmidst.getObservedChildren(variable, copyDAG);

                if(observedChildren.size() > 2) {

                    /* Iteramos por los pares de variables observadas hijas de dicha variable latente */
                    List<Tuple2<Variable, Variable>> childrenCombinations = generateVariableCombinations(observedChildren);

                    for(Tuple2<Variable, Variable> combination: childrenCombinations) {

                        /*
                            - Creamos una nueva variable latente de cardinalidad igual a la variable latente padre.
                            - Ponemos como hijos de esta nueva variable latente el par de variables observadas seleccionado (modificando el grafo)
                         */
                        Variable newLatentVar = copyVariables.newMultinomialVariable("LV_" + (this.latentVarNameCounter++), variable.getNumberOfStates());
                        copyDAG.addVariable(newLatentVar);
                        copyDAG.getParentSet(combination.getFirst()).removeParent(variable);
                        copyDAG.getParentSet(combination.getSecond()).removeParent(variable);

                        copyDAG.getParentSet(combination.getFirst()).addParent(newLatentVar);
                        copyDAG.getParentSet(combination.getSecond()).addParent(newLatentVar);
                        copyDAG.getParentSet(newLatentVar).addParent(variable);

                        /* Creamos un nuevo Plateau para el aprendizaje donde omitimos la nueva variable latente y sus hijos */
                        HashSet<Variable> omittedVariables = new HashSet<>();
                        omittedVariables.add(newLatentVar);
                        omittedVariables.addAll(GraphUtilsAmidst.getChildren(newLatentVar, copyDAG));
                        PlateuStructure copyPlateauStructure = plateuStructure.deepCopy(copyDAG, omittedVariables);

                        /* Aprendemos el modelo de forma local */
                        VBEM_Local localVBEM = new VBEM_Local(this.localVBEMConfig);
                        localVBEM.learnModel(copyPlateauStructure, copyDAG, typeLocalVBEM.variablesToUpdate(newLatentVar, copyDAG));

                        /* Comparamos el modelo generado con el mejor modelo actual */
                        if(localVBEM.getPlateuStructure().getLogProbabilityOfEvidence() > bestModelScore) {
                            bestModel = localVBEM.getPlateuStructure();
                            bestModelScore = localVBEM.getPlateuStructure().getLogProbabilityOfEvidence();
                            bestPair = combination;
                            bestPairParent = variable;
                            newLatentVarName = newLatentVar.getName(); // To avoid name discrepancies in the end
                        }

                        /* Modificamos el grafo y eliminamos el nuevo nodo latente para poder resetear el proceso */
                        copyDAG.getParentSet(combination.getFirst()).removeParent(newLatentVar);
                        copyDAG.getParentSet(combination.getSecond()).removeParent(newLatentVar);
                        copyDAG.getParentSet(newLatentVar).removeParent(variable);
                        copyDAG.removeVariable(newLatentVar);

                        copyDAG.getParentSet(combination.getFirst()).addParent(variable);
                        copyDAG.getParentSet(combination.getSecond()).addParent(variable);

                        copyVariables.remove(newLatentVar);
                    }
                }
            }
        }

        /* Si el operador produjo un mejor modelo, lo aprendemos con VBEM_HC de forma "global" */
        // TODO: Escoger el tipo de inicializacion a utilizar aqui, podria ser incluso NONE
        if(bestModelScore > -Double.MAX_VALUE) {

            // Modificamos el grafo para que no haya diferencias con la estructura del Plateau
            copyDAG.getParentSet(bestPair.getFirst()).removeParent(bestPairParent);
            copyDAG.getParentSet(bestPair.getSecond()).removeParent(bestPairParent);

            // We use "newLatentVarName" to avoid name discrepancies between the Plateau and the DAG
            Variable newLatentVar = copyVariables.newMultinomialVariable(newLatentVarName, bestPairParent.getNumberOfStates());
            copyDAG.addVariable(newLatentVar);
            copyDAG.getParentSet(bestPair.getFirst()).addParent(newLatentVar);
            copyDAG.getParentSet(bestPair.getSecond()).addParent(newLatentVar);
            copyDAG.getParentSet(newLatentVar).addParent(bestPairParent);

            if(doGlobalVBEM) {
                VBEM_Global globalVBEM = new VBEM_Global(this.globalVBEMConfig);
                globalVBEM.learnModel(bestModel, copyDAG);

                bestModel = globalVBEM.getPlateuStructure();
                bestModelScore = globalVBEM.getPlateuStructure().getLogProbabilityOfEvidence();
            }
        }

        /* Devolvemos el resultado */
        return new Result(bestModel, bestModelScore, copyDAG, "AddNode");
    }

    private List<Tuple2<Variable, Variable>> generateVariableCombinations(List<Variable> variables) {

        List<Tuple2<Variable, Variable>> variableCombinations = new ArrayList<>();
        Iterator<int[]> variableIndexCombinations = new Combinations(variables.size(), 2).iterator();

        /* Iteramos por las combinaciones no repetidas de variables observadas y generamos una Tuple con cada una */
        while(variableIndexCombinations.hasNext()) {
            // Indices de los clusters a comparar
            int[] combination = variableIndexCombinations.next();
            // Añadimos la nueva tupla
            variableCombinations.add(new Tuple2<>(variables.get(combination[0]), variables.get(combination[1])));
        }

        return variableCombinations;
    }
}
