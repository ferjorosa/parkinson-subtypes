package eu.amidst.extension.learn.structure.hillclimber.operator;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.inference.messagepassing.Node;
import eu.amidst.core.learning.parametric.bayesian.utils.PlateuIIDReplication;
import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHcConfig;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHcOperation;
import eu.amidst.extension.util.tuple.Tuple2;

import java.util.*;

public class BayesianHcRemoveArc implements BayesianHcOperator {

    /**
     * The set of arcs that need to be avoided in the structure search process. The key of the map is the tail node
     * and the List contains all the head nodes.
     */
    private Map<Variable, List<Variable>> arcBlackList;

    /** Configuration for the parameter learning process */
    private BayesianHcConfig config;

    public BayesianHcRemoveArc(BayesianHcConfig config, Map<Variable, List<Variable>> arcBlackList){
        this.config = config;
        this.arcBlackList = arcBlackList;
    }

    @Override
    public BayesianHcOperation apply(DAG dag,
                                     DataOnMemory<DataInstance> data,
                                     Map<String, double[]> priorsParameters,
                                     Map<Tuple2<Variable, Set<Variable>>, Double> scores) {

        /*
         * In order to speed up operations, we store current score of each variable, which will be used to estimate the
         * total score of each new arc.
         *
         * The scores Map is more general, as it contains all the previously evaluated arcs. We do this in order to avoid
         * creating new Tuple objects each time we want to estimate a new total score.
         */
        Map<Variable, Double> currentVarScores = new HashMap<>();
        for(Variable var: dag.getVariables()) {
            Tuple2<Variable, Set<Variable>> varTuple = new Tuple2<>(var, new HashSet<>(dag.getParentSet(var).getParents()));
            double varScore = scores.get(varTuple);
            currentVarScores.put(var, varScore);
        }

        /* Best operation that will be returned */
        BayesianHcOperation bestOperation = new BayesianHcOperation(null, null,
                -Double.MAX_VALUE, BayesianHcOperation.Type.REMOVE_ARC);

        for(Variable toVar: dag.getVariables()) {
            for (Variable fromVar : dag.getParentSet(toVar)) {
                /* Check the arc is not blacklisted */
                if (!arcBlackList.containsKey(fromVar) || !arcBlackList.get(fromVar).contains(toVar)) {

                    /* Parents minus the removed one */
                    Set<Variable> parentVars = new HashSet<>(dag.getParentSet(toVar).getParents());
                    parentVars.remove(fromVar);

                    /* Check if this operation has been previously done */
                    Tuple2<Variable, Set<Variable>> tuple = new Tuple2<>(toVar, parentVars);

                    /* If this arc has been previously estimated, sum up its total score */
                    if(scores.containsKey(tuple)) {

                        double toVarScore = scores.get(tuple);
                        double totalScore = toVarScore;

                        for(Variable var: dag.getVariables())
                            if(!var.equals(toVar))
                                totalScore += currentVarScores.get(var);

                        if(totalScore > bestOperation.getTotalScore())
                            bestOperation = new BayesianHcOperation(fromVar, toVar, totalScore,
                                    BayesianHcOperation.Type.REMOVE_ARC);
                    }
                    /* Estimate the arc score if it hasn't been previously estimated */
                    else {

                        /* Create a new Variables object with toVar's family minus fromVar */
                        List<Variable> localVars = new ArrayList<>(parentVars);
                        localVars.add(toVar);
                        Variables localVariables = new Variables(localVars);

                        /* Create a local DAG with the localVariables object */
                        DAG localDAG = new DAG(localVariables);
                        for(Variable parent: localVariables)
                            if(!parent.equals(toVar))
                                localDAG.getParentSet(toVar).addParent(parent);

                        /* Create a new Plateau object to estimate the score of the removed arc */
                        PlateuStructure localPlateau = new PlateuIIDReplication();
                        localPlateau.initTransientDataStructure();
                        localPlateau.setNRepetitions(data.getNumberOfDataInstances());
                        localPlateau.setSeed(config.getSeed());
                        localPlateau.setDAG(localDAG, priorsParameters);
                        localPlateau.replicateModel();
                        localPlateau.getVMP().setOutput(false);
                        localPlateau.getVMP().setTestELBO(false);
                        localPlateau.getVMP().setThreshold(config.getThreshold());
                        localPlateau.getVMP().setMaxIter(config.getMaxIter());
                        localPlateau.setEvidence(data.getList());

                        /*
                         * Get the list of nodes that correspond to the childVar and its parameter variables. Then
                         * estimate the parameters of this variable and its score
                         */
                        List<Node> toVarNodes = localPlateau.getNodes(toVar);
                        localPlateau.runInferenceHC(toVarNodes);
                        double toVarScore = localPlateau.getLogProbabilityOfEvidence();
                        double totalScore = toVarScore;

                        for(Variable var: dag.getVariables())
                            if(!var.equals(toVar))
                                totalScore += currentVarScores.get(var);

                        if(totalScore > bestOperation.getTotalScore())
                            bestOperation = new BayesianHcOperation(fromVar, toVar, totalScore,
                                    BayesianHcOperation.Type.REMOVE_ARC);

                        /* Add the new arc score to the Map */
                        scores.put(tuple, toVarScore);
                    }
                }
            }
        }

        return bestOperation;
    }
}
