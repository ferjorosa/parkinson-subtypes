package eu.amidst.extension.learn.structure.vbsem;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.data.DataUtils;
import eu.amidst.extension.learn.parameter.VBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHc;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHcConfig;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHcResult;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcAddArc;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcOperator;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcRemoveArc;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcReverseArc;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.PriorsFromData;
import eu.amidst.extension.util.tuple.Tuple2;

import java.util.*;

// TODO: Initialization variants similar to VBEM

/* Note: This VBSEM doesnt include (yet) the possibility of introducing arc restrictions. It will be later developed */
public class VBSEM {

    private VBEMConfig config;

    private BayesianHcConfig bayesianHcConfig;

    private int maxIterations;

    public VBSEM(VBEMConfig config, BayesianHcConfig bayesianHcConfig, int maxIterations) {
        this.config = config;
        this.bayesianHcConfig = bayesianHcConfig;
        this.maxIterations = maxIterations;
    }

    public Tuple2<BayesianNetwork, Double> learnModel(BayesianNetwork initialModel,
                                                      double initialScore,
                                                      DataOnMemory<DataInstance> data,
                                                      Set<BayesianHcOperator> hcOperators,
                                                      LogUtils.LogLevel logLevel) {

        BayesianHc bayesianHc = new BayesianHc(this.bayesianHcConfig, this.bayesianHcConfig.getMaxIter(), this.bayesianHcConfig.getThreshold(), hcOperators);

        BayesianNetwork currentModel = initialModel;
        double currentScore = initialScore;

        LogUtils.info("\nVBSEM - initial score: "+initialScore, logLevel);

        /* Create the VBEM */
        VBEM vbem = new VBEM(config);

        int iterations = 1;

        while(iterations <= maxIterations ) {

            LogUtils.info("VBSEM - iteration "+ iterations, logLevel);
            iterations++;

            /* Generate complete data using currentModel */
            DataOnMemory<DataInstance> completeData = DataUtils.completeLatentData(data, currentModel);

            /* Assign the new attributes to the latent variables in the current model */
            List<String> latentVarNames = new ArrayList<>();
            for(Variable variable: currentModel.getVariables()) {
                if (variable.getAttribute() == null) {
                    latentVarNames.add(variable.getName());
                    variable.setAttribute(completeData.getAttributes().getAttributeByName(variable.getName()));
                    variable.setObservable(true);
                }
            }

            /* Learn a new structure S[n+1] using the complete data (and Empirical Bayes priors) */
            final Map<String, double[]> priors = PriorsFromData.generate(data, 1);
            BayesianHcResult bayesianHcResult = bayesianHc.learnModel(currentModel.getDAG(), completeData, priors, logLevel);

            /* Before VBEM, remove attributes of latent variables and set them as no observable */
            for(String latentVarName: latentVarNames){
                Variable latentVar = bayesianHcResult.getDag().getVariables().getVariableByName(latentVarName);
                latentVar.setObservable(false);
                latentVar.setAttribute(null);
            }

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

    public Tuple2<BayesianNetwork, Double> learnModel(BayesianNetwork initialModel,
                                                      double initialScore,
                                                      DataOnMemory<DataInstance> data,
                                                      LogUtils.LogLevel logLevel) {

        /* Create the Bayesian Hill-climber */
        Variables variables = initialModel.getVariables();
        BayesianHcAddArc addArc = new BayesianHcAddArc(this.bayesianHcConfig, variables, 3);
        BayesianHcRemoveArc removeArc = new BayesianHcRemoveArc(this.bayesianHcConfig, new HashMap<>());
        BayesianHcReverseArc reverseArc = new BayesianHcReverseArc(this.bayesianHcConfig, variables, 3);
        Set<BayesianHcOperator> hcOperators = new LinkedHashSet<>();
        hcOperators.add(addArc);
        hcOperators.add(removeArc);
        hcOperators.add(reverseArc);

        return this.learnModel(initialModel, initialScore, data, hcOperators, logLevel);
    }
}
