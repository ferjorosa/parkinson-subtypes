package eu.amidst.extension.learn.structure.operator.incremental;

import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.VBEM_Local;
import eu.amidst.extension.learn.structure.Result;
import eu.amidst.extension.learn.structure.typelocalvbem.TypeLocalVBEM;
import eu.amidst.extension.util.GraphUtilsAmidst;
import eu.amidst.extension.util.tuple.Tuple2;
import eu.amidst.extension.util.tuple.Tuple3;

import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.PriorityQueue;
import java.util.Set;

public class BlfmIncAddDiscreteNode implements BlfmIncOperator {

    private int newNodeCardinality;

    private int maxNumberOfDiscreteLatentNodes;

    private VBEMConfig localVBEMConfig;

    private TypeLocalVBEM typeLocalVBEM;

    private int latentVarNameCounter = 0;

    public BlfmIncAddDiscreteNode(int newNodeCardinality,
                                  int maxNumberOfDiscreteLatentNodes,
                                  TypeLocalVBEM typeLocalVBEM) {
        this(newNodeCardinality, maxNumberOfDiscreteLatentNodes, new VBEMConfig(), typeLocalVBEM);
    }

    public BlfmIncAddDiscreteNode(int newNodeCardinality,
                                  int maxNumberOfDiscreteLatentNodes,
                                  VBEMConfig localVBEMConfig,
                                  TypeLocalVBEM typeLocalVBEM) {
        this.newNodeCardinality = newNodeCardinality;
        this.maxNumberOfDiscreteLatentNodes = maxNumberOfDiscreteLatentNodes;
        this.localVBEMConfig = localVBEMConfig;
        this.typeLocalVBEM = typeLocalVBEM;
    }

    /**
     *
     */
    @Override
    public Tuple3<Variable, Variable, Result> apply(Set<Variable> currentSet,
                                                    PlateuStructure plateuStructure,
                                                    DAG dag) {

        PlateuStructure bestModel = plateuStructure;
        double bestModelScore = -Double.MAX_VALUE;
        Tuple2<Variable, Variable> bestPair = null;
        String newLatentVarName = "";

        /* Return current model if current number of discrete latent nodes is maximum */
        long numberOfLatentNodes = dag.getVariables().getListOfVariables().stream().filter(x->x.isDiscrete() && !x.isObservable()).count();
        if(numberOfLatentNodes >= this.maxNumberOfDiscreteLatentNodes)
            return new Tuple3<>(null, null, new Result(bestModel, bestModelScore, dag, "AddDiscreteNode"));

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
                    Variable newLatentVar = copyVariables.newMultinomialVariable("LV_" + (this.latentVarNameCounter++), this.newNodeCardinality);
                    copyDAG.addVariable(newLatentVar);
                    copyDAG.getParentSet(firstVar).addParent(newLatentVar);
                    copyDAG.getParentSet(secondVar).addParent(newLatentVar);

                    /* Create a new plateau by copying current one and omitting the new variable and its children */
                    HashSet<Variable> omittedVariables = new HashSet<>();
                    omittedVariables.add(newLatentVar);
                    omittedVariables.addAll(GraphUtilsAmidst.getChildren(newLatentVar, copyDAG));
                    PlateuStructure copyPlateauStructure = plateuStructure.deepCopy(copyDAG, omittedVariables);

                    /* Learn the new model with Local VBEM */
                    VBEM_Local localVBEM = new VBEM_Local(this.localVBEMConfig);
                    localVBEM.learnModel(copyPlateauStructure, copyDAG, typeLocalVBEM.variablesToUpdate(newLatentVar, copyDAG));

                    /* Compare its score with current best model */
                    if (localVBEM.getPlateuStructure().getLogProbabilityOfEvidence() > bestModelScore) {
                        bestModel = localVBEM.getPlateuStructure();
                        bestModelScore = localVBEM.getPlateuStructure().getLogProbabilityOfEvidence();
                        bestPair = new Tuple2<>(firstVar, secondVar);
                        newLatentVarName = newLatentVar.getName(); // To avoid name discrepancies with the Plateau
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
            Variable newLatentVar = copyVariables.newMultinomialVariable(newLatentVarName, this.newNodeCardinality);
            copyDAG.addVariable(newLatentVar);
            copyDAG.getParentSet(bestPair.getFirst()).addParent(newLatentVar);
            copyDAG.getParentSet(bestPair.getSecond()).addParent(newLatentVar);
        }

        return new Tuple3<>(bestPair.getFirst(), bestPair.getSecond(), new Result(bestModel, bestModelScore, copyDAG, "AddDiscreteNode"));
    }

    /**
     *
     */
    @Override
    public Tuple3<Variable, Variable, Result> apply(PriorityQueue<Tuple3<Variable, Variable, Double>> selectedTriples,
                                                    PlateuStructure plateuStructure,
                                                    DAG dag) {

        PlateuStructure bestModel = plateuStructure;
        double bestModelScore = -Double.MAX_VALUE;
        Tuple2<Variable, Variable> bestPair = null;
        String newLatentVarName = "";

        /* Return current model if current number of discrete latent nodes is maximum */
        long numberOfLatentNodes = dag.getVariables().getListOfVariables().stream().filter(x->x.isDiscrete() && !x.isObservable()).count();
        if(numberOfLatentNodes >= this.maxNumberOfDiscreteLatentNodes)
            return new Tuple3<>(null, null, new Result(bestModel, bestModelScore, dag, "AddDiscreteNode"));

        /* Create a copy of the variables and DAG objects */
        Variables copyVariables = dag.getVariables().deepCopy();
        DAG copyDAG = dag.deepCopy(copyVariables);

        /* Iterate through the queue of selected triples */
        for(Tuple3<Variable, Variable, Double> triple: selectedTriples){
            Variable firstVar = copyVariables.getVariableByName(triple.getFirst().getName());
            Variable secondVar = copyVariables.getVariableByName(triple.getSecond().getName());

            /* Create a new Latent variable as the pair's new parent */
            Variable newLatentVar = copyVariables.newMultinomialVariable("LV_" + (this.latentVarNameCounter++), this.newNodeCardinality);
            copyDAG.addVariable(newLatentVar);
            copyDAG.getParentSet(firstVar).addParent(newLatentVar);
            copyDAG.getParentSet(secondVar).addParent(newLatentVar);

            /* Create a new plateau by copying current one and omitting the new variable and its children */
            HashSet<Variable> omittedVariables = new HashSet<>();
            omittedVariables.add(newLatentVar);
            omittedVariables.addAll(GraphUtilsAmidst.getChildren(newLatentVar, copyDAG));
            PlateuStructure copyPlateauStructure = plateuStructure.deepCopy(copyDAG, omittedVariables);

            /* Learn the new model with Local VBEM */
            VBEM_Local localVBEM = new VBEM_Local(this.localVBEMConfig);
            localVBEM.learnModel(copyPlateauStructure, copyDAG, typeLocalVBEM.variablesToUpdate(newLatentVar, copyDAG));

            /* Compare its score with current best model */
            if(localVBEM.getPlateuStructure().getLogProbabilityOfEvidence() > bestModelScore) {
                bestModel = localVBEM.getPlateuStructure();
                bestModelScore = localVBEM.getPlateuStructure().getLogProbabilityOfEvidence();
                bestPair = new Tuple2<>(firstVar, secondVar);
                newLatentVarName = newLatentVar.getName(); // To avoid name discrepancies with the Plateau
            }

            /* Remove the newly created node to reset the process for the next pair */
            copyDAG.getParentSet(firstVar).removeParent(newLatentVar);
            copyDAG.getParentSet(secondVar).removeParent(newLatentVar);
            copyDAG.removeVariable(newLatentVar);
            copyVariables.remove(newLatentVar);
        }

        /* Modify the DAG with the best latent var */
        if(bestModelScore > -Double.MAX_VALUE) {
            Variable newLatentVar = copyVariables.newMultinomialVariable(newLatentVarName, this.newNodeCardinality);
            copyDAG.addVariable(newLatentVar);
            copyDAG.getParentSet(bestPair.getFirst()).addParent(newLatentVar);
            copyDAG.getParentSet(bestPair.getSecond()).addParent(newLatentVar);
        }

        return new Tuple3<>(bestPair.getFirst(), bestPair.getSecond(), new Result(bestModel, bestModelScore, copyDAG, "AddDiscreteNode"));
    }
}
