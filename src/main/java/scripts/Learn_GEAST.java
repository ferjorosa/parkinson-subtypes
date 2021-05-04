package scripts;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.extension.data.DataUtils;
import eu.amidst.extension.util.LogUtils;
import methods.GEAST;
import methods.GMM;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Learning script for the Gaussian version of the EAST algorithm. It was not designed not work with missing data so we use
 * the imputed version of the dataset, which was imputed using MICE.
 *
 * [1] Poon LK, Zhang NL, Liu AH. Model-based clustering of high-dimensional data: Variable selection versus facet determination. Int J Approx Reasoning. 2013;54(1):196–215.
 * [2] Azur MJ, Stuart EA, Frangakis C, Leaf PJ. Multiple imputation by chained equations:  What is it and how does it work? Int J Methods Psychiatr Res. 2011;20(1):40–49.
 */
public class Learn_GEAST {

    public static void main(String[] args) throws Exception {
        long seed = 0;
        String settingsLocation = "geast_settings.xml";
        LogUtils.LogLevel logLevel= LogUtils.LogLevel.INFO;

        String fileName = "data/data_numerical_imputed.arff";

        learnWithImputed(seed, fileName, settingsLocation, logLevel);

    }

    private static void learnWithImputed(long seed,
                                         String fileName,
                                         String settingsLocation,
                                         LogUtils.LogLevel logLevel) throws Exception {

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

        GEAST.learnModel(data, settingsLocation, seed);
    }
}
