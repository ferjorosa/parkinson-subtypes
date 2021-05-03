package scripts;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.io.BayesianNetworkLoader;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.io.DataStreamWriter;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.extension.data.DataUtils;
import eu.amidst.extension.missing.util.ImputeMissing;

import java.util.List;
import java.util.stream.Collectors;

/** The GMM implementation doesnt allow missing data, so we impute the missing values before learning the final model.
 * We know the limitations of this course of action.
 */
public class ImputeMissingDataForGMM {

    public static void main(String[] args) throws Exception {

        String fileName = "data/data_numerical.arff";
        DataOnMemory<DataInstance> data = DataStreamLoader.loadDataOnMemoryFromFile(fileName);

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

        BayesianNetwork model = BayesianNetworkLoader.loadFromFile("results/glsl_cil_1_64.bn", data.getAttributes());

        DataOnMemory<DataInstance> dataNoMissing = ImputeMissing.imputeWithModel(data, model);
        DataStreamWriter.writeDataToFile(dataNoMissing, "data/data_numerical_imputed.arff");
    }
}
