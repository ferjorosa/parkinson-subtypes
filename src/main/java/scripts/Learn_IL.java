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
import eu.amidst.extension.data.DataUtils;
import eu.amidst.extension.io.GenieWriter;
import eu.amidst.extension.missing.util.ImputeMissing;
import eu.amidst.extension.util.EstimatePredictiveScore;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.PriorsFromData;
import eu.amidst.extension.util.tuple.Tuple2;
import methods.IL;
import scripts.util.JsonResult;

import java.io.File;
import java.io.FileWriter;
import java.io.Writer;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * The purpose of this script is to learn a latent forest using the incremental learner (IL) algorithm. For this, we
 * use the motor-nms data in its two variants:
 *
 * - With missing data.
 * - Without missing data.
 *
 * The algorithm already supports missing data imputation due to the internal use of the VBEM algorithm.
 *
 * NOTES:
 * - We can compare all models using BIC, AIC and Log-Likelihood.
 * - We can only compare models using ELBO when they have been learned using the same dataset. For example,
 *   we can compare a KDB and an LCM that have been learned using the complete data with ELBO, but we cannot compare
 *   two LCMs when one has been learned using the complete data and the other the missing data.
 */
public class Learn_IL {

    public static void main(String[] args) throws Exception {
        long seed = 0;
        String resultsPath = "results";
        int nVbemCandidates = 64;

        learnModel(seed, resultsPath, "il_high", IL.Flexibility.HIGH, nVbemCandidates, LogUtils.LogLevel.INFO);
        learnModel(seed, resultsPath, "il_medium", IL.Flexibility.MEDIUM, nVbemCandidates, LogUtils.LogLevel.INFO);
        learnModel(seed, resultsPath, "il_low", IL.Flexibility.LOW, nVbemCandidates, LogUtils.LogLevel.INFO);
    }

    private static void learnModel(long seed,
                                   String directoryPath,
                                   String fileName,
                                   IL.Flexibility flexibility,
                                   int nVbemCandidates,
                                   LogUtils.LogLevel logLevel) throws Exception {

        /* Load data with missing and remove socio-demographic variables */
        System.out.println("==================================");
        System.out.println("========== IL ==========");
        System.out.println("==================================");
        System.out.println("n VBEM candidates: " + nVbemCandidates);

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

        IL ilMethod = new IL(flexibility, seed, nVbemCandidates);

        long initTime = System.currentTimeMillis();
        Tuple2<BayesianNetwork, Double> result = ilMethod.learnModel(data, priors, logLevel);
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
         * Impute missing values with the posterior predictive, then impute latent variables values for each data instance
         * and write the resulting dataset with imputed missing and new completed latent vars
         */
        DataOnMemory<DataInstance> completeData = ImputeMissing.imputeWithModel(data, posteriorPredictive);
        DataOnMemory<DataInstance> completeDataWithLatents = DataUtils.completeLatentData(completeData, posteriorPredictive);
        DataStreamWriter.writeDataToFile(completeDataWithLatents, directoryPath + "/" + fileName + ".arff");

        /* Write the XDSL (Genie) file */
        DataUtils.defineAttributesMaxMinValues(completeDataWithLatents);
        GenieWriter genieWriter = new GenieWriter();
        genieWriter.write(posteriorPredictive, new File(directoryPath + "/" + fileName + ".xdsl"));

        /* Export model in AMIDST format */
        BayesianNetworkWriter.save(posteriorPredictive, directoryPath + "/" + fileName + ".bn");
    }
}
