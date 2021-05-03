package methods;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.extension.util.tuple.Tuple4;
import methods.util.AmidstToLatlabData;
import org.latlab.core.data.MixedDataSet;
import org.latlab.core.learner.geast.GeastWithoutPouch;
import org.latlab.core.learner.geast.IModelWithScore;
import org.latlab.core.learner.geast.ParameterGenerator;
import org.latlab.core.learner.geast.Settings;
import org.latlab.core.model.Builder;
import org.latlab.core.model.Gltm;
import org.latlab.core.util.DiscreteVariable;

import java.text.DecimalFormatSymbols;
import java.util.Locale;

public class GEAST {

    private String settingsLocation;
    private long seed;
    private String dataNameAttribute;

    public GEAST(String settingsLocation,
                 long seed) {
        this.settingsLocation = settingsLocation;
        this.seed = seed;
    }

    public static Tuple4<IModelWithScore, Double, Double, Long> learnModel(DataOnMemory<DataInstance> trainData,
                                                                           String settingsLocation,
                                                                           long seed) throws Exception {

        System.out.println("\n==========================");
        System.out.println("GEAST");
        System.out.println("==========================\n");

        MixedDataSet trainDataLatlab = AmidstToLatlabData.transform(trainData);

        Settings settings = new Settings(settingsLocation, trainDataLatlab, trainDataLatlab.name());
        GeastWithoutPouch geastWithoutPouch = settings.createGeastWithoutPouch(seed);

        long initTime = System.currentTimeMillis();
        Gltm lcm =  Builder.buildNaiveBayesModel(
                new Gltm(),
                new DiscreteVariable(2),
                trainDataLatlab.getNonClassVariables());
        ParameterGenerator parameterGenerator = new ParameterGenerator(trainDataLatlab);
        parameterGenerator.generate(lcm);
        IModelWithScore result = geastWithoutPouch.learn(lcm);
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
}
