package methods;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.tuple.Tuple4;
import methods.util.AmidstToLatlabData;
import org.latlab.core.data.MixedDataSet;
import org.latlab.core.learner.geast.Geast;
import org.latlab.core.learner.geast.IModelWithScore;
import org.latlab.core.learner.geast.Settings;
import org.latlab.core.model.Builder;
import org.latlab.core.model.Gltm;
import org.latlab.core.util.Variable;

import java.text.DecimalFormatSymbols;
import java.util.List;
import java.util.Locale;

public class GMM {

    private String settingsLocation;
    private long seed;

    public GMM(String settingsLocation, long seed) {
        this.settingsLocation = settingsLocation;
        this.seed = seed;
    }

    public static Tuple4<IModelWithScore, Double, Double, Long> learnModel(DataOnMemory<DataInstance> trainData,
                                                                           String settingsLocation,
                                                                           LogUtils.LogLevel logLevel,
                                                                           long seed) throws Exception {

        System.out.println("\n==========================");
        System.out.println("GMM");
        System.out.println("==========================\n");

        MixedDataSet trainDataLatlab = AmidstToLatlabData.transform(trainData);
        Settings settings = new Settings(settingsLocation, trainDataLatlab, trainDataLatlab.name());
        Geast geast = settings.createGeast(seed);

        long initTime = System.currentTimeMillis();
        IModelWithScore result = learntoMaxCardinality(geast, trainDataLatlab.getNonClassVariables(), logLevel);
        long endTime = System.currentTimeMillis();
        long learningTimeMs = (endTime - initTime);
        double learningTimeS = learningTimeMs / 1000;

        DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
        otherSymbols.setDecimalSeparator('.');
        System.out.println("\n---------------------------------------------");
        System.out.println("Log-Likelihood: " + result.loglikelihood());
        System.out.println("BIC: " + result.BicScore());
        System.out.println("Learning time (ms): " + learningTimeMs + " ms");
        System.out.println("Learning time (s): " + learningTimeS + " s");

        return new Tuple4<>(result, result.loglikelihood(), result.BicScore(), learningTimeMs);
    }

    private static IModelWithScore learntoMaxCardinality(Geast geast, List<Variable> nonClassVariables, LogUtils.LogLevel logLevel) {

        IModelWithScore bestResult = null;
        double bestBIC = -Double.MAX_VALUE;

        for(int card = 2; card < Integer.MAX_VALUE; card++) {

            long initTime = System.currentTimeMillis();
            Gltm gmm = Builder.buildMixedMixtureModel(
                    new Gltm(),
                    card,
                    nonClassVariables);

            geast.context().parameterGenerator().generate(gmm);
            IModelWithScore result = geast.context().estimationEm().estimate(gmm);
            long endTime = System.currentTimeMillis();
            long learnTime = (endTime - initTime);

            double currentBic = result.BicScore();

            LogUtils.info("\nCardinality " + card, logLevel);
            LogUtils.info("BIC: " +  currentBic, logLevel);
            LogUtils.info("Log-likelihood: " + result.loglikelihood(), logLevel);
            LogUtils.info("Time: " + learnTime + " ms", logLevel);

            if(currentBic > bestBIC) {
                bestResult = result;
                bestBIC = currentBic;
            } else {
                LogUtils.info("SCORE STOPPED IMPROVING", logLevel);
                return bestResult;
            }
        }

        return bestResult;
    }
}
