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
import eu.amidst.extension.util.GraphUtilsAmidst;
import eu.amidst.extension.util.tuple.Tuple2;

import java.util.*;

public class BayesianHcReverseArc implements BayesianHcOperator {

    /**
     * The set of arcs that need to be avoided in the structure search process. The key of the map is the tail node
     * and the List contains all the head nodes.
     */
    private Map<Variable, List<Variable>> arcBlackList;

    /** Maximum number of parent nodes. */
    private Map<Variable, Integer> maxNumberOfParents;

    /** Configuration for the parameter learning process */
    private BayesianHcConfig config;

    /** Constructor where we establish everything */
    public BayesianHcReverseArc(BayesianHcConfig config,
                                Variables variables,
                                Map<Variable, List<Variable>> arcBlackList,
                                Map<Variable, Integer> maxNumberOfParents) {

        this.config = config;
        this.arcBlackList = arcBlackList;
        this.maxNumberOfParents = maxNumberOfParents;

        /* Introduce default arc restrictions */
        for(Variable var1: variables)
            for(Variable var2: variables)
                if(var1.isContinuous() && var2.isDiscrete() && !this.arcBlackList.get(var1).contains(var2))
                    this.arcBlackList.get(var1).add(var2);
    }

    /** Constructor that sets a default number of max parents for each variable */
    public BayesianHcReverseArc(BayesianHcConfig config,
                                Variables variables,
                                Map<Variable, List<Variable>> arcBlackList,
                                int maxNumberOfParents){
        this.config = config;
        this.arcBlackList = arcBlackList;
        this.maxNumberOfParents = new HashMap<>();

        /* Create Map of max number of parents */
        for(Variable var: variables)
            this.maxNumberOfParents.put(var, maxNumberOfParents);

        /* Introduce default arc restrictions */
        for(Variable var1: variables)
            for(Variable var2: variables)
                if(var1.isContinuous() && var2.isDiscrete() && !this.arcBlackList.get(var1).contains(var2))
                    this.arcBlackList.get(var1).add(var2);
    }

    /** Constructor that creates arc restrictions from data (discrete variable cannot have a continuous parent)
     * and sets a default number of max parents for each variable */
    public BayesianHcReverseArc(BayesianHcConfig config,
                                Variables variables,
                                int maxNumberOfParents) {

        this.config = config;
        this.maxNumberOfParents = new HashMap<>();

        /* Create Map of max number of parents */
        for(Variable var: variables)
            this.maxNumberOfParents.put(var, maxNumberOfParents);

        /* Initialize the arc black list */
        this.arcBlackList = new HashMap<>();
        for(Variable var: variables)
            this.arcBlackList.put(var, new ArrayList<>());

        /* Introduce arc restrictions */
        for(Variable var1: variables)
            for(Variable var2: variables)
                if(var1.isContinuous() && var2.isDiscrete())
                    this.arcBlackList.get(var1).add(var2);
    }

    public Map<Variable, List<Variable>> getArcBlackList() {
        return arcBlackList;
    }

    public Map<Variable, Integer> getMaxNumberOfParents() {
        return maxNumberOfParents;
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
        BayesianHcOperation bestOperation = new BayesianHcOperation(null, null, -Double.MAX_VALUE, BayesianHcOperation.Type.REVERSE_ARC);

        /* CopyDAG created to check if a reverse arc is allowed */
        DAG dagForCheckingAllowedArcs = dag.deepCopy(dag.getVariables());

        /* Iteration through current arcs */
        for(Variable toVar: dag.getVariables()) {
            for (Variable fromVar : dag.getParentSet(toVar)) {
                /* Check the arc is not blacklisted */
                if ((!arcBlackList.containsKey(toVar) || !arcBlackList.get(toVar).contains(fromVar))
                        && isArcReverseAllowed(dagForCheckingAllowedArcs, fromVar, toVar)) {

                    /* Estimate the reversed arc score */
                    double removeArcScore = removeArc(dag, fromVar, toVar, data, priorsParameters, currentVarScores, scores);
                    double addArcScore = addArc(dag, toVar, fromVar, data, priorsParameters, currentVarScores, scores); // New reversed arc
                    double totalScore = removeArcScore + addArcScore;

                    /* Combine the reversed arc score with the scores of the rest of the arcs */
                    for(Variable var: currentVarScores.keySet())
                        if(!var.equals(toVar) && !var.equals(fromVar))
                            totalScore += currentVarScores.get(var);

                    if(totalScore > bestOperation.getTotalScore())
                        bestOperation = new BayesianHcOperation(fromVar, toVar, totalScore, BayesianHcOperation.Type.REVERSE_ARC);
                }
            }
        }

        return bestOperation;
    }

    private boolean isArcReverseAllowed(DAG dag, Variable fromVar, Variable toVar) {

        /* Reverse the arc in the dag and see if cycles are formed */
        dag.getParentSet(toVar).removeParent(fromVar);
        dag.getParentSet(fromVar).addParent(toVar);

        boolean isAllowed = true;
        if(GraphUtilsAmidst.containsPath(fromVar, toVar, dag))
            isAllowed = false;
        if(dag.getParentSet(fromVar).getNumberOfParents() > this.maxNumberOfParents.get(fromVar))
            isAllowed = false;

        /* Reverse the arc back to its original direction */
        dag.getParentSet(fromVar).removeParent(toVar);
        dag.getParentSet(toVar).addParent(fromVar);

        return isAllowed;
    }

    private double addArc(DAG dag,
                          Variable fromVar,
                          Variable toVar,
                          DataOnMemory<DataInstance> data,
                          Map<String, double[]> priorsParameters,
                          Map<Variable, Double> currentVarScores,
                          Map<Tuple2<Variable, Set<Variable>>, Double> scores) {

        Set<Variable> parentVars = new HashSet<>(dag.getParentSet(toVar).getParents());
        parentVars.add(fromVar);

        /* Check if this operation has been previously done */
        Tuple2<Variable, Set<Variable>> tuple = new Tuple2<>(toVar, parentVars);

        /* If this arc has been previously estimated, sum up its total score */
        if(scores.containsKey(tuple)) {

            double toVarScore = scores.get(tuple);
            double totalScore = toVarScore;

            for(Variable var: dag.getVariables())
                if(!var.equals(toVar))
                    totalScore += currentVarScores.get(var);

            return totalScore;
        }
        /* Estimate the arc score if it hasn't been previously estimated */
        else {

            /* Create a new Variables object containing the variables involved in the new arc */
            List<Variable> localVars = new ArrayList<>(parentVars);
            localVars.add(toVar);
            Variables localVariables = new Variables(localVars);

            /* Create a local DAG that contains existing parent arcs and the new one */
            DAG localDAG = new DAG(localVariables);
            for (Variable parent : dag.getParentSet(toVar))
                localDAG.getParentSet(toVar).addParent(parent);
            localDAG.getParentSet(toVar).addParent(fromVar);

            /* Create a new Plateau object to estimate the score of the new arc */
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

            for (Variable var : dag.getVariables())
                if (!var.equals(toVar))
                    totalScore += currentVarScores.get(var);

            /* Add the new arc score to the Map */
            scores.put(tuple, toVarScore);

            return totalScore;
        }
    }

    private double removeArc(DAG dag,
                             Variable fromVar,
                             Variable toVar,
                             DataOnMemory<DataInstance> data,
                             Map<String, double[]> priorsParameters,
                             Map<Variable, Double> currentVarScores,
                             Map<Tuple2<Variable, Set<Variable>>, Double> scores) {

        Set<Variable> parentVars = new HashSet<>(dag.getParentSet(toVar).getParents());
        parentVars.remove(fromVar);

        /* Check if this operation has been previously done */
        Tuple2<Variable, Set<Variable>> tuple = new Tuple2<>(toVar, parentVars);

        /* If this arc has been previously estimated, sum up its total score */
        if(scores.containsKey(tuple)) {

            double toVarScore = scores.get(tuple);
            double totalScore = toVarScore;

            for (Variable var : dag.getVariables())
                if (!var.equals(toVar))
                    totalScore += currentVarScores.get(var);

            return totalScore;
        }
        /* Estimate the arc score if it hasn't been previously estimated */
        else {

            /* Create a new Variables object containing the variables involved in the new arc */
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

            /* Add the new arc score to the Map */
            scores.put(tuple, toVarScore);

            return totalScore;
        }
    }
}
