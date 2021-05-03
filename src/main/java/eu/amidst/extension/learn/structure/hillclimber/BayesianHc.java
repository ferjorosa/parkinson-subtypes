package eu.amidst.extension.learn.structure.hillclimber;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.learning.parametric.bayesian.utils.PlateuIIDReplication;
import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcOperator;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.tuple.Tuple2;

import java.util.*;

// TODO: Remmeber that in HC is it necessary to update the edgeBlackList once a new arc has been added/removed OUT OF IT

/**
 * We use dynamic programming to store the scores of all the evaluated arcs. We can do this thanks to score decomposition.
 */
public class BayesianHc {

    private BayesianHcConfig config;

    private int maxHcIter;

    private double thresholdHc;

    private Set<BayesianHcOperator> operators;

    public BayesianHc(BayesianHcConfig config,
                      int maxHcIter,
                      double thresholdHc,
                      Set<BayesianHcOperator> operators) {
        this.config = config;
        this.maxHcIter = maxHcIter;
        this.thresholdHc = thresholdHc;
        this.operators = operators;
    }

    public Set<BayesianHcOperator> getOperators() {
        return operators;
    }

    public BayesianHcResult learnModel(DAG dag,
                                       DataOnMemory<DataInstance> data,
                                       Map<String, double[]> priorsParameters,
                                       LogUtils.LogLevel logLevel) {

        /* Copy the DAG */
        Variables copyVariables = dag.getVariables().deepCopy();
        DAG copyDAG = dag.deepCopy(copyVariables);

        /* Learn the initial model parameters and estimate its global score and its score Map */
        PlateuStructure plateuStructure = learnModelParameters(copyDAG, data, priorsParameters);
        Map<Tuple2<Variable, Set<Variable>>, Double> scores = plateuStructure.computeLogProbabilityOfEvidenceMap_3();
        double currentScore = plateuStructure.getLogProbabilityOfEvidence();

        LogUtils.info("HC - initial score: " + currentScore, logLevel);

        int iterations = 0;
        while(iterations < this.maxHcIter) {
            iterations = iterations + 1;

            /* Select the best operation (highest score) for current DAG */
            BayesianHcOperation bestOperation = selectBestOperation(copyDAG, data, priorsParameters, scores);
            LogUtils.info("Best operation = " + bestOperation.getType() +
                    "(" + bestOperation.getFromVar() + " , " + bestOperation .getToVar()+ ") -> " + bestOperation.getTotalScore(), logLevel);

            /* Compare the score of the best operation with current score. If there isn't enough improvement, STOP */
            if(currentScore >= bestOperation.getTotalScore() || Math.abs(bestOperation.getTotalScore() - currentScore) < this.thresholdHc) {
                plateuStructure = learnModelParameters(copyDAG, data, priorsParameters);
                plateuStructure.updateParameterVariablesPrior(plateuStructure.getParameterVariablesPosterior());
                BayesianNetwork bn = new BayesianNetwork(copyDAG, plateuStructure.getEFLearningBN().toConditionalDistribution());
                return new BayesianHcResult(currentScore, bn, "score ceased to increase");
            }

            /* Apply the best operation to current DAG and update currentScore */
            currentScore = performOperation(bestOperation, copyDAG);
        }

        /* Return current DAG and score if the maximum number of iterations have been reached without the score ceasing to increase */
        plateuStructure = learnModelParameters(copyDAG, data, priorsParameters);
        plateuStructure.updateParameterVariablesPrior(plateuStructure.getParameterVariablesPosterior());
        BayesianNetwork bn = new BayesianNetwork(copyDAG, plateuStructure.getEFLearningBN().toConditionalDistribution());
        return new BayesianHcResult(currentScore, bn, "maxIter reached");
    }

    /** Apply all operators and select the best operation from all of them */
    private BayesianHcOperation selectBestOperation(DAG dag,
                                                    DataOnMemory<DataInstance> data,
                                                    Map<String, double[]> priorsParameters,
                                                    Map<Tuple2<Variable, Set<Variable>>, Double> scores){

        List<BayesianHcOperation> operations = new ArrayList<>();

        /* Apply operators and store all their results */
        for(BayesianHcOperator operator: this.operators)
            operations.add(operator.apply(dag, data, priorsParameters, scores));

        /* If the list is empty, return a "fake" operation with a negative infinity score */
        if(operations.isEmpty())
            return new BayesianHcOperation(null, null, -Double.MAX_VALUE, BayesianHcOperation.Type.ADD_ARC);

        /* If it is not empty, order the list and return the best operation (should be the first one) */
        Comparator<BayesianHcOperation> byScore = (operation1, operation2) -> Double.compare(operation1.getTotalScore(), operation2.getTotalScore());
        operations.sort(Collections.reverseOrder(byScore));

        return operations.get(0);
    }

    private double performOperation(BayesianHcOperation operation, DAG dag) {

        switch (operation.getType()) {
            case ADD_ARC:
                dag.getParentSet(operation.getToVar()).addParent(operation.getFromVar());
                break;

            case REMOVE_ARC:
                dag.getParentSet(operation.getToVar()).removeParent(operation.getFromVar());
                break;

            case REVERSE_ARC:
                dag.getParentSet(operation.getToVar()).removeParent(operation.getFromVar());
                dag.getParentSet(operation.getFromVar()).addParent(operation.getToVar());
                break;
        }

        return operation.getTotalScore();
    }

    private PlateuStructure learnModelParameters(DAG dag,
                                                 DataOnMemory<DataInstance> data,
                                                 Map<String, double[]> priorsParameters) {

        PlateuStructure plateuStructure = new PlateuIIDReplication();
        plateuStructure.initTransientDataStructure();
        plateuStructure.setNRepetitions(data.getNumberOfDataInstances());
        plateuStructure.setSeed(this.config.getSeed());
        plateuStructure.setDAG(dag, priorsParameters);
        plateuStructure.replicateModel();
        plateuStructure.getVMP().setOutput(false);
        plateuStructure.getVMP().setTestELBO(false);
        plateuStructure.getVMP().setThreshold(this.config.getThreshold());
        plateuStructure.getVMP().setMaxIter(this.config.getMaxIter());
        plateuStructure.setEvidence(data.getList());

        plateuStructure.runInference();

        return plateuStructure;
    }
}
