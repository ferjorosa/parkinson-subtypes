package scripts;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.io.BayesianNetworkWriter;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.io.DataStreamWriter;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.data.DataUtils;
import eu.amidst.extension.io.GenieWriter;
import eu.amidst.extension.learn.parameter.InitializationTypeVBEM;
import eu.amidst.extension.learn.parameter.InitializationVBEM;
import eu.amidst.extension.learn.parameter.VBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.penalizer.BishopPenalizer;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHc;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHcConfig;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHcResult;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcAddArc;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcOperator;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcRemoveArc;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcReverseArc;
import eu.amidst.extension.learn.structure.vbsem.InitializationTypeVBSEM;
import eu.amidst.extension.learn.structure.vbsem.InitializationVBSEM;
import eu.amidst.extension.missing.VBSEM_Complete;
import eu.amidst.extension.missing.util.ImputeMissing;
import eu.amidst.extension.util.EstimatePredictiveScore;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.PriorsFromData;
import eu.amidst.extension.util.tuple.Tuple2;
import scripts.util.JsonResult;

import java.io.File;
import java.io.FileWriter;
import java.io.Writer;
import java.util.*;
import java.util.stream.Collectors;

/**
 * The purpose of this script is to learn a Bayesian network without latent variables. There are two variants:
 *
 * - With missing data, we use the VBSEM algorithm, which internally uses the Bayesian HC algorithm.
 * - With complete data, we directly use the the Bayesian HC algorithm, no need to impute missing values.
 *
 * NOTES:
 * - We can compare all models using BIC, AIC and Log-Likelihood.
 * - We can only compare models using ELBO when they have been learned using the same dataset. For example,
 *   we can compare a KDB and an LCM that have been learned using the complete data with ELBO, but we cannot compare
 *   two LCMs when one has been learned using the complete data and the other the missing data.
 */
public class Learn_HC {

    public static void main(String[] args) throws Exception {

        long seed = 0;
        String resultsPath = "results";
        int nVbsemCandidates = 1;
        int nVbemCandidates = 64;

        learnModel(seed, resultsPath, "hc_" +nVbsemCandidates + "_" + nVbemCandidates, nVbsemCandidates, nVbemCandidates, LogUtils.LogLevel.INFO, LogUtils.LogLevel.INFO);
    }

    /** Learn a BN without latent variables when missing data is present using the VBSEM algorithm */
    // Note: It doesnt use InitializationVBSEM because it uses VBSEM_Complete not VBSEM_restrictions
    private static void learnModel(long seed,
                                         String directoryPath,
                                         String fileName,
                                         int nVbemCandidates,
                                         int nVbsemCandidates,
                                         LogUtils.LogLevel vbsemLogLevel,
                                         LogUtils.LogLevel hcLogLevel) throws Exception {

        /* Load data with missing and remove socio-demographic variables */
        System.out.println("==================================");
        System.out.println("========== HC ==========");
        System.out.println("==================================");
        System.out.println("n VBEM candidates: " + nVbemCandidates);
        System.out.println("n VBSEM candidates: " + nVbsemCandidates);

        DataOnMemory<DataInstance> data = DataStreamLoader.loadDataOnMemoryFromFile("data/data_numerical.arff");

        /* Filter the socio-demographic attributes */
        List<Attribute> symptomAttributes = data.getAttributes().getFullListOfAttributes().stream()
                .filter(x->
                        !x.getName().equals("patnum") &&
                                !x.getName().equals("age") &&
                                !x.getName().equals("sex") &&
                                !x.getName().equals("pdonset") &&
                                !x.getName().equals("durat_pd") &&
                                !x.getName().equals("hy")
                        )
                .collect(Collectors.toList());
        data = DataUtils.project(data, symptomAttributes);

        /* Generate Empirical Bayes priors from data, ignoring missing values */
        Map<String, double[]> priors = PriorsFromData.generate(data, 1);

        InitializationVBEM initializationVBEM = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, nVbemCandidates, nVbemCandidates/2, false);
        VBEMConfig vbemConfig = new VBEMConfig(seed, 0.01, 100, initializationVBEM, new BishopPenalizer());
        BayesianHcConfig bayesianHcConfig = new BayesianHcConfig(seed, 0.01, 100);
        InitializationVBSEM initializationVBSEM = new InitializationVBSEM(InitializationTypeVBSEM.PYRAMID, nVbsemCandidates, nVbsemCandidates/2, 0.2, true);

        Variables variables = new Variables(data.getAttributes());
        DAG baseDAG = new DAG(variables);

        BayesianHcAddArc bayesianHcAddArc = new BayesianHcAddArc(bayesianHcConfig, variables, 3);
        BayesianHcRemoveArc bayesianHcRemoveArc = new BayesianHcRemoveArc(bayesianHcConfig, new HashMap<>());
        BayesianHcReverseArc bayesianHcReverseArc = new BayesianHcReverseArc(bayesianHcConfig, variables, 3);
        Set<BayesianHcOperator> bayesianHcOperators = new LinkedHashSet<>(3);
        bayesianHcOperators.add(bayesianHcAddArc);
        bayesianHcOperators.add(bayesianHcRemoveArc);
        bayesianHcOperators.add(bayesianHcReverseArc);

        long initTime = System.currentTimeMillis();
        VBEM vbem = new VBEM(vbemConfig);
        double baseScore = vbem.learnModelWithPriorUpdate(data, baseDAG);
        BayesianNetwork baseBN = vbem.getLearntBayesianNetwork();

        VBSEM_Complete vbsem_complete = new VBSEM_Complete(vbemConfig, bayesianHcConfig, 100);
        Tuple2<BayesianNetwork,Double> result = vbsem_complete.learnModel(baseBN, baseScore, data, bayesianHcOperators, vbsemLogLevel);
        long endTime = System.currentTimeMillis();

        BayesianNetwork posteriorPredictive = result.getFirst();
        double elbo = result.getSecond();
        double logLikelihood = EstimatePredictiveScore.amidstLL(posteriorPredictive, data);
        double bic = EstimatePredictiveScore.amidstBIC(posteriorPredictive, data);
        double aic = EstimatePredictiveScore.amidstAIC(posteriorPredictive, data);
        double learningTime = (endTime - initTime) / 1000.0;

        System.out.println("---------------------");
        System.out.println("Learning time (seconds): " + learningTime);
        System.out.println("ELBO: " + elbo);
        System.out.println("LogLikelihood: " + logLikelihood);
        System.out.println("BIC: " + bic);
        System.out.println("AIC: " + aic);

        /* Write the Json result file */
        JsonResult jsonResult = new JsonResult(learningTime, elbo, logLikelihood, bic, aic,0, nVbemCandidates);
        try (Writer writer = new FileWriter(directoryPath + "/" + fileName + ".json")) {
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            gson.toJson(jsonResult, writer);
        }

        /*
         * Impute missing values with the posterior predictive, no latent vars are present in this model
         */
        DataOnMemory<DataInstance> completeData = ImputeMissing.imputeWithModel(data, posteriorPredictive);
        DataStreamWriter.writeDataToFile(completeData, directoryPath + "/" + fileName + ".arff");

        /* Write the XDSL (Genie) file */
        DataUtils.defineAttributesMaxMinValues(completeData);
        GenieWriter genieWriter = new GenieWriter();
        genieWriter.write(posteriorPredictive, new File(directoryPath + "/" + fileName + ".xdsl"));

        /* Export model in AMIDST format */
        BayesianNetworkWriter.save(posteriorPredictive, directoryPath + "/" + fileName + ".bn");
    }
}
