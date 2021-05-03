package scripts;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.io.BayesianNetworkLoader;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.extension.data.DataUtils;
import eu.amidst.extension.util.EstimatePredictiveScore;

import java.util.List;
import java.util.stream.Collectors;

public class LoadModel {

    public static void main(String[] args) throws Exception {
        DataOnMemory<DataInstance> data = DataStreamLoader.loadDataOnMemoryFromFile("data/data_missing_categorical.arff");
        DataOnMemory<DataInstance> data_imputed = DataStreamLoader.loadDataOnMemoryFromFile("data/data_missing_categorical_complete.arff");

        /* Filter the socio-demographic attributes */
        List<Attribute> symptomAttributes = data.getAttributes().getFullListOfAttributes().stream()
                .filter(x->
                        !x.getName().equals("patnum") &&
                                !x.getName().equals("age") &&
                                !x.getName().equals("sex") &&
                                !x.getName().equals("pdonset") &&
                                !x.getName().equals("durat_pd"))
                .collect(Collectors.toList());
        data = DataUtils.project(data, symptomAttributes);

        //BayesianNetwork model = BayesianNetworkLoader.loadFromFile("results/missing/il_low.bn");
        BayesianNetwork model = BayesianNetworkLoader.loadFromFile("results/missing/il_glsl.bn", data.getAttributes());

        double logLikelihood = EstimatePredictiveScore.amidstLL(model, data_imputed);
        double bic = EstimatePredictiveScore.amidstBIC(model, data_imputed);
        double aic = EstimatePredictiveScore.amidstAIC(model, data_imputed);

        double logLikelihood_2 = EstimatePredictiveScore.amidstLL(model, data);
        double bic_2 = EstimatePredictiveScore.amidstBIC(model, data);
        double aic_2 = EstimatePredictiveScore.amidstAIC(model, data);

        System.out.println("LogLikelihood: " + logLikelihood);
        System.out.println("BIC: " + bic);
        System.out.println("AIC: " + aic);

        System.out.println("LogLikelihood: " + logLikelihood_2);
        System.out.println("BIC: " + bic_2);
        System.out.println("AIC: " + aic_2);
    }
}
