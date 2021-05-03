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
import eu.amidst.extension.learn.parameter.InitializationTypeVBEM;
import eu.amidst.extension.learn.parameter.InitializationVBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.penalizer.BishopPenalizer;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHcConfig;
import eu.amidst.extension.learn.structure.vbsem.InitializationTypeVBSEM;
import eu.amidst.extension.learn.structure.vbsem.InitializationVBSEM;
import eu.amidst.extension.missing.util.ImputeMissing;
import eu.amidst.extension.util.EstimatePredictiveScore;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.PriorsFromData;
import eu.amidst.extension.util.tuple.Tuple2;
import methods.KDB;
import scripts.util.JsonResult;

import java.io.File;
import java.io.FileWriter;
import java.io.Writer;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * The purpose of this script is to learn and export a KDB model of the combined motor-nms data for its comparison with
 * the rest of the methods. We learn two variants (and then compare them):
 *
 * - With missing data, using the internal VBSEM to complete the data with a latent KDB structure.
 * - With complete data, after using the VBSEM (network without latent vars) to complete missing values.
 *
 * NOTES:
 * - We can compare all models using BIC, AIC and Log-Likelihood.
 * - We can only compare models using ELBO when they have been learned using the same dataset. For example,
 *   we can compare a KDB and an LCM that have been learned using the complete data with ELBO, but we cannot compare
 *   two LCMs when one has been learned using the complete data and the other the missing data.
 */
// TODO: inicializacion con NONE ahora mismo
public class Learn_KDB {

    public static void main(String[] args) throws Exception {
        long seed = 0;
        String resultsPath = "results";
        int nVbsemCandidates = 1;
        int nVbemCandidates = 64;

        learnModel(seed, resultsPath, "kdb_" +nVbsemCandidates + "_" + nVbemCandidates, nVbemCandidates, nVbsemCandidates, LogUtils.LogLevel.INFO, LogUtils.LogLevel.DEBUG, LogUtils.LogLevel.INFO);
    }

    /** Learn a KDB using data with missing values */
    private static void learnModel(long seed,
                                   String directoryPath,
                                   String fileName,
                                   int nVbemCandidates,
                                   int nVbsemCandidates,
                                   LogUtils.LogLevel kdbLogLevel,
                                   LogUtils.LogLevel vbsemLogLevel,
                                   LogUtils.LogLevel hcLogLevel) throws Exception {

        /* Load data with missing and remove socio-demographic variables */
        System.out.println("==================================");
        System.out.println("========== KDB =========");
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
        InitializationVBSEM initializationVBSEM = new InitializationVBSEM(InitializationTypeVBSEM.NONE, nVbsemCandidates, nVbsemCandidates/2, 0.2, true);
        KDB kdbMethod = new KDB(vbemConfig, bayesianHcConfig, 100, initializationVBSEM);

        long initTime = System.currentTimeMillis();
        Tuple2<BayesianNetwork, Double> result = kdbMethod.learnModelIteratively(data, 2, 100, priors, kdbLogLevel, vbsemLogLevel, hcLogLevel);
        long endTime = System.currentTimeMillis();

        BayesianNetwork posteriorPredictive = result.getFirst();
        double elbo = result.getSecond();
        double logLikelihood = EstimatePredictiveScore.amidstLL(posteriorPredictive, data);
        double bic = EstimatePredictiveScore.amidstBIC(posteriorPredictive, data);
        double aic = EstimatePredictiveScore.amidstAIC(posteriorPredictive, data);
        double learningTime = (endTime - initTime) / 1000.0;

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
