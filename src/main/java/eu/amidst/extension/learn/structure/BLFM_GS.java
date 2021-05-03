package eu.amidst.extension.learn.structure;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.learn.parameter.VBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.VBEM_Local;
import eu.amidst.extension.learn.structure.operator.hc.tree.BltmHcDecreaseCard;
import eu.amidst.extension.learn.structure.operator.hc.tree.BltmHcIncreaseCard;
import eu.amidst.extension.learn.structure.typelocalvbem.TypeLocalVBEM;
import eu.amidst.extension.util.GraphUtilsAmidst;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.tuple.Tuple2;
import eu.amidst.extension.util.tuple.Tuple3;

import java.util.*;

public class BLFM_GS {

    private VBEMConfig vbemConfig;

    private TypeLocalVBEM typeLocalVBEM;

    private int latentVarNameCounter = 0;

    public BLFM_GS(VBEMConfig vbemConfig, TypeLocalVBEM typeLocalVBEM) {
        this.vbemConfig = vbemConfig;
        this.typeLocalVBEM = typeLocalVBEM;
    }

    public Result learnModel(DataOnMemory<DataInstance> data,
                             Map<String, double[]> priors,
                             LogUtils.LogLevel logLevel) {


        /* Inicializamos las estructuras necesarias */
        Variables variables = new Variables(data.getAttributes());
        DAG dag = new DAG(variables);

        Set<Variable> currentSet = new LinkedHashSet<>(); // Current set of variables being considered
        for(Variable variable: variables)
            currentSet.add(variable);

        /* Aprendemos el modelo inicial donde todas las variables son independientes y no hay latentes */
        VBEM vbem = new VBEM(this.vbemConfig);
        double initialScore = vbem.learnModel(data, dag, priors);
        Result bestResult = new Result(vbem.getPlateuStructure(), initialScore, dag, "BLFM_GS");

        LogUtils.info("Initial score: " + bestResult.getElbo(), logLevel);

        boolean keepsImproving = true;
        int iteration = 0;
        while(keepsImproving && currentSet.size() > 1) {

            iteration++;
            List<Variable> partitionVariables = new ArrayList<>();

            /* 1 - Create a new partition and compare the score with current best model */
            Tuple3<Variable, Variable, Result> newPartitionResult = createPartition(currentSet, bestResult.getPlateuStructure(), bestResult.getDag());
            if(newPartitionResult.getThird().getElbo() > bestResult.getElbo()) {
                /* If it improves it, update current best model, remove vars from currentSet and create new partition */
                currentSet.remove(newPartitionResult.getFirst());
                currentSet.remove(newPartitionResult.getSecond());
                partitionVariables.add(newPartitionResult.getFirst());
                partitionVariables.add(newPartitionResult.getSecond());
                bestResult = newPartitionResult.getThird();
                LogUtils.info("New partition: (" + newPartitionResult.getFirst().getName() + "," + newPartitionResult.getSecond().getName()+")", logLevel);
                LogUtils.info("Score: " + newPartitionResult.getThird().getElbo(), logLevel);
            } else {
                /* If not, return current best model */
                return bestResult;
            }

            /* 2 - Expand the partition one attribute at a time until it stops improving the score */
            boolean keepExpansion = true;
            while(keepExpansion) {
                Tuple2<Variable, Result> partitionExpansion = expandPartition(partitionVariables, currentSet, bestResult.getPlateuStructure(), bestResult.getDag());
                if(partitionExpansion.getSecond().getElbo() > bestResult.getElbo()) {
                    currentSet.remove(partitionExpansion.getFirst());
                    partitionVariables.add(partitionExpansion.getFirst());
                    bestResult = partitionExpansion.getSecond();
                    LogUtils.info("Expansion: " + partitionExpansion.getFirst().getName(), logLevel);
                    LogUtils.info("Score: " + partitionExpansion.getSecond().getElbo(), logLevel);
                } else
                    keepExpansion = false;
            }
            LogUtils.info("\n", logLevel);
        }

        return bestResult;
    }

    /*
     * Se evalua cada par de variables del currentset y se crea una particion con cada uno de ellos.
     * La cardinalidad del modelo se estima para cada caso.
     */
    private Tuple3<Variable, Variable, Result> createPartition(Set<Variable> currentSet,
                                                               PlateuStructure plateuStructure,
                                                               DAG dag) {
        PlateuStructure bestModel = plateuStructure;
        double bestModelScore = -Double.MAX_VALUE;
        int bestModelCardinality = -1;
        Tuple2<Variable, Variable> bestPair = null;
        String newLatentVarName = "";

        /* Create a copy of the variables and DAG objects */
        Variables copyVariables = dag.getVariables().deepCopy();
        DAG copyDAG = dag.deepCopy(copyVariables);
        /* Copy currentSet */
        Set<Variable> copyCurrentSet = new LinkedHashSet<>();
        for(Variable var: currentSet)
            copyCurrentSet.add(copyVariables.getVariableByName(var.getName()));

        /* Iterate through the list of variable pairs that belong to currentSet */
        for(Variable firstVar: copyCurrentSet) {
            for (Variable secondVar : copyCurrentSet) {
                if (!firstVar.equals(secondVar)) {
                    /* Create a new Latent variable as the pair's new parent */
                    Variable newLatentVar = copyVariables.newMultinomialVariable("LV_" + (this.latentVarNameCounter++), 2);
                    copyDAG.addVariable(newLatentVar);
                    copyDAG.getParentSet(firstVar).addParent(newLatentVar);
                    copyDAG.getParentSet(secondVar).addParent(newLatentVar);

                    /* Create a new plateau by copying current one and omitting the new variable and its children */
                    HashSet<Variable> omittedVariables = new HashSet<>();
                    omittedVariables.add(newLatentVar);
                    omittedVariables.addAll(GraphUtilsAmidst.getChildren(newLatentVar, copyDAG));
                    PlateuStructure copyPlateauStructure = plateuStructure.deepCopy(copyDAG, omittedVariables);

                    /* Learn the new model with Local VBEM */
                    VBEM_Local localVBEM = new VBEM_Local(this.vbemConfig);
                    localVBEM.learnModel(copyPlateauStructure, copyDAG, typeLocalVBEM.variablesToUpdate(newLatentVar, copyDAG));

                    /* Locally estimate the cardinality of the partition LV */
                    Result cardinalityResult = estimateLocalCardinality(newLatentVarName, copyDAG, localVBEM.getPlateuStructure());

                    /* Compare its score with current best model */
                    if(cardinalityResult.getElbo() > bestModelScore) {
                        bestModel = cardinalityResult.getPlateuStructure();
                        bestModelScore = cardinalityResult.getElbo();
                        bestPair = new Tuple2<>(firstVar, secondVar);
                        newLatentVarName = newLatentVar.getName(); // To avoid name discrepancies with the Plateau
                        bestModelCardinality = cardinalityResult.getDag().getVariables().getVariableByName(newLatentVarName).getNumberOfStates(); // To produce the same LV cardinality
                    }

                    /* Remove the newly created node to reset the process for the next pair */
                    copyDAG.getParentSet(firstVar).removeParent(newLatentVar);
                    copyDAG.getParentSet(secondVar).removeParent(newLatentVar);
                    copyDAG.removeVariable(newLatentVar);
                    copyVariables.remove(newLatentVar);
                }
            }
        }

        /* Modify the DAG with the best latent var */
        if(bestModelScore > -Double.MAX_VALUE) {
            Variable newLatentVar = copyVariables.newMultinomialVariable(newLatentVarName, bestModelCardinality);
            copyDAG.addVariable(newLatentVar);
            copyDAG.getParentSet(bestPair.getFirst()).addParent(newLatentVar);
            copyDAG.getParentSet(bestPair.getSecond()).addParent(newLatentVar);
        }

        return new Tuple3<>(bestPair.getFirst(), bestPair.getSecond(), new Result(bestModel, bestModelScore, copyDAG, "CreatePartition"));
    }

    /*
     * Se evalua cada variable del currentSet junto con la particion actual.
     * La cardinalidad del modelo se estima para cada caso.
     */
    private Tuple2<Variable, Result> expandPartition(List<Variable> partitionVariables,
                                                    Set<Variable> currentSet,
                                                    PlateuStructure plateuStructure,
                                                    DAG dag) {
        PlateuStructure bestModel = plateuStructure;
        double bestModelScore = -Double.MAX_VALUE;
        int bestModelCardinality = -1;
        Variable bestVar = null;

        /* Create a copy of the variables and DAG objects */
        Variables copyVariables = dag.getVariables().deepCopy();
        DAG copyDAG = dag.deepCopy(copyVariables);
        Set<Variable> copyCurrentSet = new LinkedHashSet<>();
        for(Variable var: currentSet)
            copyCurrentSet.add(copyVariables.getVariableByName(var.getName()));

        /* Get the partition LV */
        Variable partitionLV = copyDAG.getParentSet(copyVariables.getVariableByName(partitionVariables.get(0).getName())).getParents().get(0);

        /* Iterate through each of the current set variables */
        for(Variable var: copyCurrentSet) {

            /* Modify the DAG to add the variable to the partition */
            copyDAG.getParentSet(var).addParent(partitionLV);

            /* Create a new plateau by copying current one and omitting the partition variables + var */
            HashSet<Variable> omittedVariables = new HashSet<>(partitionVariables);
            omittedVariables.add(var);
            PlateuStructure copyPlateauStructure = plateuStructure.deepCopy(copyDAG, omittedVariables);

            /* Learn the parameters of the new partition with local VBEM */
            VBEM_Local localVBEM = new VBEM_Local(this.vbemConfig);
            localVBEM.learnModel(copyPlateauStructure, copyDAG, typeLocalVBEM.variablesToUpdate(partitionLV, copyDAG));

            /* Locally estimate the cardinality of the partition LV */
            Result cardinalityResult = estimateLocalCardinality(partitionLV.getName(), copyDAG, localVBEM.getPlateuStructure());

            /* Compare its score with current best model */
            if (cardinalityResult.getElbo() > bestModelScore) {
                bestModel = cardinalityResult.getPlateuStructure();
                bestModelScore = cardinalityResult.getElbo();
                bestVar = var;
                bestModelCardinality = cardinalityResult.getDag().getVariables().getVariableByName(partitionLV.getName()).getNumberOfStates(); // To produce the same LV cardinality
            }

            /* Remove the newly created arc to reset the process for the next pair */
            copyDAG.getParentSet(var).removeParent(partitionLV);
        }

        /* Modify the DAG with the best arc */
        if(bestModelScore > -Double.MAX_VALUE) {
            copyDAG.getParentSet(bestVar).addParent(partitionLV);
            partitionLV.setNumberOfStates(bestModelCardinality);
            return new Tuple2<>(bestVar, new Result(bestModel, bestModelScore, copyDAG, "ExpandPartition"));
        }

        return new Tuple2<>(null, new Result(bestModel, bestModelScore, copyDAG, "ExpandPartition"));
    }

    private Result estimateLocalCardinality(String partitionLvName, DAG dag, PlateuStructure plateuStructure) {

        List<String> discreteLatentVars = new ArrayList<>(1);
        discreteLatentVars.add(partitionLvName);

        int maxCardinality = Integer.MAX_VALUE;
        BltmHcIncreaseCard increaseCardOperator = new BltmHcIncreaseCard(maxCardinality, this.vbemConfig, this.vbemConfig, typeLocalVBEM);
        BltmHcDecreaseCard decreaseCardOperator = new BltmHcDecreaseCard(2, this.vbemConfig, this.vbemConfig, typeLocalVBEM);

        Result bestResult = new Result(plateuStructure, plateuStructure.getLogProbabilityOfEvidence(), dag, "EstimateLocalCardinality");

        while (true) {
            Result increaseCardResult = increaseCardOperator.apply(bestResult.getPlateuStructure(), bestResult.getDag(), discreteLatentVars,  false);
            Result decreaseCardResult = decreaseCardOperator.apply(bestResult.getPlateuStructure(), bestResult.getDag(), discreteLatentVars, false);

            if(increaseCardResult.getElbo() > decreaseCardResult.getElbo() && increaseCardResult.getElbo() > bestResult.getElbo())
                bestResult = increaseCardResult;
            else if(decreaseCardResult.getElbo() > increaseCardResult.getElbo() && decreaseCardResult.getElbo() > bestResult.getElbo())
                bestResult = decreaseCardResult;
            else
                return bestResult;
        }
    }
}
