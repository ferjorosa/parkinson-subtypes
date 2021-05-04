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
import eu.amidst.extension.learn.structure.glsl.GLSL;
import eu.amidst.extension.learn.structure.glsl.operator.*;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHcConfig;
import eu.amidst.extension.learn.structure.vbsem.InitializationTypeVBSEM;
import eu.amidst.extension.learn.structure.vbsem.InitializationVBSEM;
import eu.amidst.extension.missing.util.ImputeMissing;
import eu.amidst.extension.util.EstimatePredictiveScore;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.PriorsFromData;
import eu.amidst.extension.util.tuple.Tuple2;
import methods.CIL;
import methods.IL;
import scripts.util.JsonResult;

import java.io.File;
import java.io.FileWriter;
import java.io.Writer;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Learning script for the greedy latent structure learner (GLSL) algorithm. This method is dependent on the starting point
 * so we tried several possibilities:
 *
 *   - Empty graph.
 *   - Incremental learner result.
 *   - Constrained incremental learner result. We tried alpha values of 1 and 10. They resulted in identical models.
 *
 * GLSL can work with missing data due to its internal Variational structural EM algorithm.
 */
public class Learn_GLSL {

    public static void main(String[] args) throws Exception {

        long seed = 0;
        String directoryPath = "results/";
        int nVbemCandidates = 64;
        int nVbsemCandidates = 1;
        int maxNumerParents_latent = 3;
        int maxNumberParents_observed = 1;
        IL.Flexibility il_flexibility = IL.Flexibility.HIGH;
        CIL.Flexibility cil_flexibility = CIL.Flexibility.HIGH;

        //learnWithEmpty(seed, directoryPath, "glsl_" + nVbsemCandidates + "_" + nVbemCandidates, maxNumerParents_latent, maxNumberParents_observed, nVbemCandidates, nVbsemCandidates);
        //learnWithIL(seed, directoryPath, "glsl_il_" + nVbsemCandidates + "_" + nVbemCandidates, maxNumerParents_latent, maxNumberParents_observed, il_flexibility, nVbemCandidates, nVbsemCandidates, LogUtils.LogLevel.INFO);
        leanWithCIL(seed, directoryPath, "glsl_cil_" + nVbsemCandidates + "_" + nVbemCandidates, maxNumerParents_latent, maxNumberParents_observed, cil_flexibility, 10, nVbemCandidates, nVbsemCandidates, LogUtils.LogLevel.INFO);
    }

    private static void learnWithEmpty(long seed,
                                       String directoryPath,
                                       String fileName,
                                       int maxNumberParents_latent,
                                       int maxNumberParents_observed,
                                       int nVbemCandidates,
                                       int nVbsemCandidates) throws Exception {

        /* Load data with missing and remove socio-demographic variables */
        System.out.println("==================================");
        System.out.println("========== GLSL (empty) ==========");
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

        /**************************************************************************************************************/
        /* Generate the empty network (all variables are independent and there are no latent vars) and learn its parameters */
        Variables variables = new Variables(data.getAttributes());
        DAG emptyDag = new DAG(variables);
        VBEM vbem = new VBEM();
        double emptyDagScore = vbem.learnModelWithPriorUpdate(data, emptyDag);
        BayesianNetwork emptyBn = vbem.getLearntBayesianNetwork();
        Tuple2<BayesianNetwork, Double> resultEmpty = new Tuple2<>(emptyBn, emptyDagScore);

        /**************************************************************************************************************/

        InitializationVBEM initializationVBEM = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, nVbemCandidates, nVbemCandidates/2, true);
        VBEMConfig vbemConfig = new VBEMConfig(seed, 0.01, 100, initializationVBEM, new BishopPenalizer());
        InitializationVBSEM initializationVBSEM = new InitializationVBSEM(InitializationTypeVBSEM.NONE, nVbsemCandidates, nVbsemCandidates/2, 0.2, true);
        BayesianHcConfig bayesianHcConfig = new BayesianHcConfig(seed, 0.01, 100);

        GLSL_IncreaseCard glsl_increaseCard = new GLSL_IncreaseCard(Integer.MAX_VALUE, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM);
        GLSL_DecreaseCard glsl_decreaseCard = new GLSL_DecreaseCard(2, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM);
        GLSL_RemoveLatent glsl_removeLatent = new GLSL_RemoveLatent(maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM);
        GLSL_AddLatent_child glsl_addLatent_child = new GLSL_AddLatent_child(Integer.MAX_VALUE, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM, 60000);
        GLSL_AddLatent_ind glsl_addLatent_ind = new GLSL_AddLatent_ind(Integer.MAX_VALUE, 2, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM, 120000);
        Set<GLSL_Operator> glslOperators = new LinkedHashSet<>(2);
        glslOperators.add(glsl_increaseCard);
        glslOperators.add(glsl_decreaseCard);
        glslOperators.add(glsl_removeLatent);
        glslOperators.add(glsl_addLatent_child);
        glslOperators.add(glsl_addLatent_ind);

        long initTime = System.currentTimeMillis();
        GLSL glsl = new GLSL(Integer.MAX_VALUE, glslOperators);
        Tuple2<BayesianNetwork, Double> result = glsl.learnModel(resultEmpty.getFirst(), resultEmpty.getSecond(), data, LogUtils.LogLevel.INFO, LogUtils.LogLevel.INFO);
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

    private static void leanWithCIL(long seed,
                                    String directoryPath,
                                    String fileName,
                                    int maxNumberParents_latent,
                                    int maxNumberParents_observed,
                                    CIL.Flexibility flexibility,
                                    int alpha,
                                    int nVbemCandidates,
                                    int nVbsemCandidates,
                                    LogUtils.LogLevel cilLogLevel) throws Exception {

        /* Load data with missing and remove socio-demographic variables */
        System.out.println("==================================");
        System.out.println("========== GLSL with CIL ==========");
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

        /**************************************************************************************************************/
        /* Run the CIL method */
        CIL cilMethod = new CIL(flexibility, seed, nVbemCandidates);

        long initTimeCIL = System.currentTimeMillis();
        Tuple2<BayesianNetwork, Double> resultCIL = cilMethod.learnModel(data, alpha, priors, cilLogLevel);
        long endTimeCIL = System.currentTimeMillis();

        long learningTimeCIL = endTimeCIL - initTimeCIL;

        System.out.println("Time CIL: " + learningTimeCIL);

        /**************************************************************************************************************/

        InitializationVBEM initializationVBEM = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, nVbemCandidates, nVbemCandidates/2, true);
        VBEMConfig vbemConfig = new VBEMConfig(seed, 0.01, 100, initializationVBEM, new BishopPenalizer());
        InitializationVBSEM initializationVBSEM = new InitializationVBSEM(InitializationTypeVBSEM.NONE, nVbsemCandidates, nVbsemCandidates/2, 0.2, true);
        BayesianHcConfig bayesianHcConfig = new BayesianHcConfig(seed, 0.01, 100);

        GLSL_IncreaseCard glsl_increaseCard = new GLSL_IncreaseCard(Integer.MAX_VALUE, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM);
        GLSL_DecreaseCard glsl_decreaseCard = new GLSL_DecreaseCard(2, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM);
        GLSL_RemoveLatent glsl_removeLatent = new GLSL_RemoveLatent(maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM);
        GLSL_AddLatent_child glsl_addLatent_child = new GLSL_AddLatent_child(Integer.MAX_VALUE, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM, 60000);
        GLSL_AddLatent_ind glsl_addLatent_ind = new GLSL_AddLatent_ind(Integer.MAX_VALUE, 2, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM, 120000);
        Set<GLSL_Operator> glslOperators = new LinkedHashSet<>(2);
        glslOperators.add(glsl_increaseCard);
        glslOperators.add(glsl_decreaseCard);
        glslOperators.add(glsl_removeLatent);
        glslOperators.add(glsl_addLatent_child);
        glslOperators.add(glsl_addLatent_ind);

        long initTime = System.currentTimeMillis();
        GLSL glsl = new GLSL(Integer.MAX_VALUE, glslOperators);
        Tuple2<BayesianNetwork, Double> result = glsl.learnModel(resultCIL.getFirst(), resultCIL.getSecond(), data, LogUtils.LogLevel.INFO, LogUtils.LogLevel.INFO);
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

    private static void learnWithIL(long seed,
                                    String directoryPath,
                                    String fileName,
                                    int maxNumberParents_latent,
                                    int maxNumberParents_observed,
                                    IL.Flexibility flexibility,
                                    int nVbemCandidates,
                                    int nVbsemCandidates,
                                    LogUtils.LogLevel ilLogLevel) throws Exception {

        /* Load data with missing and remove socio-demographic variables */
        System.out.println("==================================");
        System.out.println("========== GLSL with IL ==========");
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

        /**************************************************************************************************************/
        /* Run the IL method */
        IL ilMethod = new IL(flexibility, seed, nVbemCandidates);

        long initTimeIL = System.currentTimeMillis();
        Tuple2<BayesianNetwork, Double> resultIL = ilMethod.learnModel(data, priors, ilLogLevel);
        long endTimeIL = System.currentTimeMillis();

        long learningTimeIL = endTimeIL - initTimeIL;

        System.out.println("Time IL: " + learningTimeIL);

        /**************************************************************************************************************/

        InitializationVBEM initializationVBEM = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, nVbemCandidates, nVbemCandidates/2, true);
        VBEMConfig vbemConfig = new VBEMConfig(seed, 0.01, 100, initializationVBEM, new BishopPenalizer());
        InitializationVBSEM initializationVBSEM = new InitializationVBSEM(InitializationTypeVBSEM.NONE, nVbsemCandidates, nVbsemCandidates/2, 0.2, true);
        BayesianHcConfig bayesianHcConfig = new BayesianHcConfig(seed, 0.01, 100);

        GLSL_IncreaseCard glsl_increaseCard = new GLSL_IncreaseCard(Integer.MAX_VALUE, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM);
        GLSL_DecreaseCard glsl_decreaseCard = new GLSL_DecreaseCard(2, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM);
        GLSL_RemoveLatent glsl_removeLatent = new GLSL_RemoveLatent(maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM);
        GLSL_AddLatent_child glsl_addLatent_child = new GLSL_AddLatent_child(Integer.MAX_VALUE, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM, 60000);
        GLSL_AddLatent_ind glsl_addLatent_ind = new GLSL_AddLatent_ind(Integer.MAX_VALUE, 2, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM, 120000);
        Set<GLSL_Operator> glslOperators = new LinkedHashSet<>(2);
        glslOperators.add(glsl_increaseCard);
        glslOperators.add(glsl_decreaseCard);
        glslOperators.add(glsl_removeLatent);
        glslOperators.add(glsl_addLatent_child);
        glslOperators.add(glsl_addLatent_ind);

        long initTime = System.currentTimeMillis();
        GLSL glsl = new GLSL(Integer.MAX_VALUE, glslOperators);
        Tuple2<BayesianNetwork, Double> result = glsl.learnModel(resultIL.getFirst(), resultIL.getSecond(), data, LogUtils.LogLevel.INFO, LogUtils.LogLevel.INFO);
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
