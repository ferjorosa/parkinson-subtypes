package eu.amidst.extension.util;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.extension.data.DataUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * If the variable is categorical, estimate its frequencies that will be the parameters of the dirichlet.
 * If the variable is continuous, estimate its mean and precision.
 */
public class PriorsFromData {

    /** In this method we assume a constant number of pseudocounts for all prior parameters, independently if it is a discrete
     * or continuous variable.
     *
     * Note: We work with String instead of Attribute / Variable objects to make it easier to work with
     */
    public static Map<String, double[]> generate(DataOnMemory<DataInstance> data, int pseudocounts) {

        Map<String, double[]> priorsMap = new HashMap<>();

        for(Attribute attribute: data.getAttributes()){
            List<Attribute> attributes = new ArrayList<>();
            attributes.add(attribute);
            DataOnMemory<DataInstance> projectedData = DataUtils.project(data, attributes);

            /* We first use a List to ignore NaN values */
            List<Double> projectedDataList = new ArrayList<>(projectedData.getNumberOfDataInstances());
            for(int i=0; i < projectedData.getNumberOfDataInstances(); i++){
                if(!Double.isNaN(projectedData.getDataInstance(i).toArray()[0]))
                    projectedDataList.add(projectedData.getDataInstance(i).toArray()[0]);
            }
            double[] projectedDataArray = projectedDataList.stream().mapToDouble(x->x).toArray();

            if(attribute.isDiscrete()) {
                /* Estimate frequencies */
                double[] frequencies = estimateFrequencies(projectedDataArray, attribute.getNumberOfStates());

                /* Multiply by the psudocounts */
                for(int i=0; i < frequencies.length; i++)
                    frequencies[i] = frequencies[i] * pseudocounts;

                /* Store the result in the Map */
                priorsMap.put(attribute.getName(), frequencies);

            } else if(attribute.isContinuous()) {
                /* Estimate mean and stDev */
                double[] meanAndPrecision = estimateMeanAndPrecision(projectedDataArray);

                /*
                * Generate Normal-Gamma parameters (https://en.wikipedia.org/wiki/Normal-gamma_distribution#Interpretation_of_parameters)
                * where we assume the same number of pseudocounts for the mean and precision.
                *
                * These parameters will be internally transformed into natural form
                */
                // TODO: This is the one proposed by Masegosa in his code , where lambda (wikipedia's notation) is represented by precision
                double mean = meanAndPrecision[0];
                double precision = meanAndPrecision[1];
                double alpha = pseudocounts / 2.0;
                double beta = alpha / precision;
                /*
                double[] normalGammaParameters = new double[4];
                normalGammaParameters[0] = mean;
                normalGammaParameters[1] = precision;
                normalGammaParameters[2] = alpha;
                normalGammaParameters[3] = beta;
                priorsMap.put(attribute.getName(), normalGammaParameters);
                */
                // TODO: A second option which I interpret from wikipedia is the following:
                /* (https://en.wikipedia.org/wiki/Normal-gamma_distribution#Interpretation_of_parameters) */
                double[] normalGammaParameters = new double[4];
                normalGammaParameters[0] = mean;
                normalGammaParameters[1] = pseudocounts;
                normalGammaParameters[2] = pseudocounts / 2.0;
                normalGammaParameters[3] = pseudocounts / (2.0 * precision);
                priorsMap.put(attribute.getName(), normalGammaParameters);

            }
        }

        return priorsMap;
    }

    private static double[]  estimateFrequencies(double[] projectedDataArray, int numberOfStates) {

        double[] counts = new double[numberOfStates];
        for(int i=0; i < projectedDataArray.length; i++) {
            int value = (int) projectedDataArray[i];
            counts[value]++;
        }

        /* Laplace smoothing */
        int laplaceCounts = 1;
        for(int i=0; i < counts.length; i++)
            counts[i] = counts[i] + laplaceCounts;

        int totalNumberOfInstances = projectedDataArray.length + (counts.length * laplaceCounts);
        double[] frequencies = new double[numberOfStates];
        for(int i=0; i < counts.length; i++)
            frequencies[i] = counts[i] / totalNumberOfInstances;

        return frequencies;
    }

    private static double[] estimateMeanAndPrecision(double[] projectedDataArray) {

        double mean = MyMath.mean(projectedDataArray);
        double stDev = MyMath.stDev(projectedDataArray);
        double precision = 1 / Math.pow(stDev, 2);

        double[] meanAndPrecision = new double[2];
        meanAndPrecision[0] = mean;
        meanAndPrecision[1] = precision;

        return meanAndPrecision;
    }

}
