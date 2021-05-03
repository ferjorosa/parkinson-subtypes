package eu.amidst.extension.learn.structure.glsl;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.variables.Variable;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.structure.glsl.operator.GLSL_Operator;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHcConfig;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcAddArc;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcOperator;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcRemoveArc;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcReverseArc;
import eu.amidst.extension.learn.structure.vbsem.InitializationVBSEM;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.tuple.Tuple2;
import eu.amidst.extension.util.tuple.Tuple3;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Greedy latent structure learner (GLSL)
 *
 * This method learns the latent structure of data using a greedy method that applies the following operators:
 *
 *  - AddLatent. Introduce a latent variable by selecting two variables. This operator usually has restrictions, such as
 *   children should have no other parents or should not have a number of parents.
 *  - RemoveLatent. Remove a latent variable and thus the arcs from it to the children variables.
 *  - IncreaseCard. Increase the cardinality of a latent variable by 1.
 *  - DecreaseCard. Decrease the cardinality of a latent variable by 1.
 *
 *  Each operator applies a VBSEM to modifiy the structure. This is also interesting for missing data.
 *
 *  We define two types of maximum parents to easily allow different types of structures. For example, when both
 *  observed variables and latent variables have a maximum number of parents of 1, the resulting structure would
 *  be a tree/forest. If the number is higher than 1 it can be a polytre/polyforest or a DAG
 *
 *  NOTE: Careful when introducing restrictions into the HC as new latent variables may be introduced or old ones
 *        be removed
 */
// TODO: We introduce default arc restrictions in the VBSEM:
//  - no latent var can have an observed parent
//  - no categorical var can have a gaussian parent
public class GLSL {

    private int maxIterations;

    private Set<GLSL_Operator> glsl_operators;

    public GLSL(int maxIterations,
                Set<GLSL_Operator> glsl_operators){
        this.maxIterations = maxIterations;
        this.glsl_operators = glsl_operators;
    }

    public Tuple2<BayesianNetwork, Double> learnModel(BayesianNetwork baseModel,
                                                      double baseScore,
                                                      DataOnMemory<DataInstance> data,
                                                      LogUtils.LogLevel iterationLogLevel,
                                                      LogUtils.LogLevel operatorLogLevel) {

        /* Base model */
        Tuple2<BayesianNetwork, Double> currentResult = new Tuple2<>(baseModel, baseScore);

        LogUtils.info("Iteration 0: " + currentResult.getSecond(), iterationLogLevel);

        /* Main loop. Iterate through the set of operators, selecting the best and compare the result's score with the current best*/
        int iterations = 0;
        while (iterations < this.maxIterations) {
            iterations += 1;
            LogUtils.info("\nIteration " + iterations + ":", iterationLogLevel);
            Tuple3<String, BayesianNetwork, Double> iterationResult = new Tuple3<>("no_operator", null, -Double.MAX_VALUE);

            /* Iterate through the set of GLSL operators */
            for(GLSL_Operator glsl_operator: glsl_operators) {
                Tuple3<String, BayesianNetwork, Double> result = glsl_operator.apply(currentResult.getFirst().getDAG(), data, operatorLogLevel);
                if(result.getThird() == -Double.MAX_VALUE)
                    LogUtils.info("\n" + result.getFirst() + " -> NONE", iterationLogLevel);
                else
                    LogUtils.info(result.getFirst() + " -> " + result.getThird(), iterationLogLevel);

                if(result.getThird() > iterationResult.getThird()) {
                    iterationResult = result;
                }
            }

            LogUtils.info("\nBest operator: "+iterationResult.getFirst(), iterationLogLevel);

            /* In case the new iteration result doesnt improve current result, stop the algorithm and return current result */
            if(iterationResult.getThird() <= currentResult.getSecond()) {
                LogUtils.info("Doesn't improve the score: " + iterationResult.getThird() + " <= " + currentResult.getSecond() + " (old best)", iterationLogLevel);
                LogUtils.info("--------------------------------------------------", iterationLogLevel);
                return currentResult;
            }

            LogUtils.info("Improves the score: " + iterationResult.getThird() + " > " + currentResult.getSecond() + " (old best)", iterationLogLevel);
            LogUtils.info("--------------------------------------------------", iterationLogLevel);
            currentResult = new Tuple2<>(iterationResult.getSecond(), iterationResult.getThird());
        }

        return currentResult;
    }
}
