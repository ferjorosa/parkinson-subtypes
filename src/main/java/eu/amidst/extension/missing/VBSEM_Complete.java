package eu.amidst.extension.missing;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.extension.learn.parameter.VBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHc;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHcConfig;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHcResult;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcOperator;
import eu.amidst.extension.missing.util.ImputeMissing;
import eu.amidst.extension.missing.util.LocateMissing;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.PriorsFromData;
import eu.amidst.extension.util.tuple.Tuple2;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * VBSEM version where data is completed using hard imputation (VMP) before applying the HC on each iteration
 *
 * TODO: compare this version with the non-complete-data version, and analyze which one is theoretically and practically better.
 */
public class VBSEM_Complete {

    private VBEMConfig config;

    private BayesianHcConfig bayesianHcConfig;

    private int maxSemIterations;

    public VBSEM_Complete(VBEMConfig config, BayesianHcConfig bayesianHcConfig, int maxSemIterations) {
        this.config = config;
        this.bayesianHcConfig = bayesianHcConfig;
        this.maxSemIterations = maxSemIterations;
    }

    public Tuple2<BayesianNetwork, Double> learnModel(BayesianNetwork initialModel,
                                                      double initialScore,
                                                      DataOnMemory<DataInstance> data,
                                                      Set<BayesianHcOperator> hcOperators,
                                                      LogUtils.LogLevel logLevel) {

        /* Initialization */
        BayesianHc bayesianHc = new BayesianHc(this.bayesianHcConfig, 1000, 0.05, hcOperators);
        VBEM vbem = new VBEM(config);
        BayesianNetwork currentModel = initialModel;
        double currentScore = initialScore;
        LinkedHashMap<Integer, List<Attribute>> missingValueLocations = LocateMissing.locateMissingValues(data);

        LogUtils.info("\nVBSEM - initial score: " + initialScore, logLevel);


        int iterations = 1;
        while (iterations <= maxSemIterations) {
            LogUtils.info("VBSEM - iteration "+ iterations, logLevel);
            iterations++;

            /* Impute missing data using current model */
            DataOnMemory<DataInstance> completeData = ImputeMissing.imputeWithModel(data, missingValueLocations, currentModel);

            /* Learn a new structure S[n+1] using the complete data (and Empirical Bayes priors) */
            final Map<String, double[]> priors = PriorsFromData.generate(completeData, 1);
            BayesianHcResult bayesianHcResult = bayesianHc.learnModel(currentModel.getDAG(), completeData, priors, logLevel);

            /* Learn the true parameters Theta[n+1] of the structure S[n+1] */
            double newScore = vbem.learnModelWithPriorUpdate(data, bayesianHcResult.getDag(), priors);
            BayesianNetwork newModel = vbem.getLearntBayesianNetwork();

            /* Update the current model if the score has increased. If not, return the current model */
            if(newScore <= currentScore || Math.abs(newScore - currentScore) < config.threshold()) {
                LogUtils.info("VBSEM - Failed to improve model: "+ newScore, logLevel);
                return new Tuple2<>(currentModel, currentScore);
            }

            LogUtils.info("VBSEM - Model updated: " + newScore, logLevel);
            currentModel = newModel;
            currentScore = newScore;
        }

        /* In case we have surpassed the maximum number of iterations, return current model and score */
        return new Tuple2<>(currentModel, currentScore);
    }
}
