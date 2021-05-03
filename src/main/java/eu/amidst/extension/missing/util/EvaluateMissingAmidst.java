package eu.amidst.extension.missing.util;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.variables.stateSpaceTypes.RealStateSpace;
import eu.amidst.extension.data.DataUtils;

import java.util.List;
import java.util.stream.Collectors;

public class EvaluateMissingAmidst {

    /** Return the accuracy of discrete missing values imputation */
    public static double accuracy(DataOnMemory<DataInstance> dataWithMissing,
                                  DataOnMemory<DataInstance> trueData,
                                  DataOnMemory<DataInstance> imputedData) {

        List<Integer> discreteAttributesIndexes = dataWithMissing.getAttributes().getFullListOfAttributes().stream()
                .filter(Attribute::isDiscrete)
                .map(Attribute::getIndex)
                .collect(Collectors.toList());

        double nMissingValues = 0;
        double correctMissingValues = 0;
        for(int i = 0; i<dataWithMissing.getNumberOfDataInstances(); i++) {
            for(int j: discreteAttributesIndexes) {
                if(Double.isNaN(dataWithMissing.getDataInstance(i).toArray()[j])) {
                    nMissingValues++;
                    if(imputedData.getDataInstance(i).toArray()[j] == trueData.getDataInstance(i).toArray()[j])
                        correctMissingValues++;
                }
            }
        }

        if(nMissingValues == 0)
            return 1.0;

        return correctMissingValues/nMissingValues;
    }

    /** Return the mean squared error of continuous missing values imputation */
    public static double mse(DataOnMemory<DataInstance> dataWithMissing,
                             DataOnMemory<DataInstance> trueData,
                             DataOnMemory<DataInstance> imputedData) {

        List<Integer> continuousAttributesIndexes = dataWithMissing.getAttributes().getFullListOfAttributes().stream()
                .filter(Attribute::isContinuous)
                .map(Attribute::getIndex)
                .collect(Collectors.toList());

        double nMissingValues = 0;
        double totalError = 0;
        double error = 0;
        for(int i = 0; i < dataWithMissing.getNumberOfDataInstances(); i++) {
            for(int j: continuousAttributesIndexes) {
                if(Double.isNaN(dataWithMissing.getDataInstance(i).toArray()[j])) {
                    nMissingValues++;
                    error = imputedData.getDataInstance(i).toArray()[j] - trueData.getDataInstance(i).toArray()[j];
                    totalError += Math.pow(error, 2);
                }
            }
        }

        if(nMissingValues == 0)
            return 0.0;

        return totalError / nMissingValues;
    }

    /**
     * Return the normalized root mean squared error of the imputed continuous values
     * see: https://en.wikipedia.org/wiki/Root-mean-square_deviation#Normalized_root-mean-square_deviation
     */
    public static double nrmse(DataOnMemory<DataInstance> dataWithMissing,
                               DataOnMemory<DataInstance> trueData,
                               DataOnMemory<DataInstance> imputedData) {

        List<Integer> continuousAttributesIndexes = dataWithMissing.getAttributes().getFullListOfAttributes().stream()
                .filter(Attribute::isContinuous)
                .map(Attribute::getIndex)
                .collect(Collectors.toList());

        /* Estimate the maximum and minimum value of each attribute (for error normalization) */
        DataUtils.defineAttributesMaxMinValues(trueData);

        /* Estimate the normalized error of imputed values */
        double normalizedError = 0.0;
        for(int j: continuousAttributesIndexes) {
            double attributeMissingValues = 0.0;
            double attributeError = 0.0;
            for(int i = 0; i < dataWithMissing.getNumberOfDataInstances(); i++) {
                if(Double.isNaN(dataWithMissing.getDataInstance(i).toArray()[j])) {
                    attributeMissingValues++;
                    double error = imputedData.getDataInstance(i).toArray()[j] - trueData.getDataInstance(i).toArray()[j];
                    attributeError += Math.pow(error, 2);
                }
            }
            attributeError = attributeError / attributeMissingValues;
            attributeError = Math.sqrt(attributeError);
            double attributeMax = ((RealStateSpace) trueData.getAttributes().getFullListOfAttributes().get(j).getStateSpaceType()).getMaxInterval();
            double attributeMin = ((RealStateSpace) trueData.getAttributes().getFullListOfAttributes().get(j).getStateSpaceType()).getMinInterval();
            double normalizedAttributeError = attributeError / (attributeMax - attributeMin);
            normalizedError += normalizedAttributeError;
        }

        return normalizedError / continuousAttributesIndexes.size();
    }

    /** Returns the average error of discrete and continuous attributes.
     *  - Continuos: NRMSE
     *  - Categorical: Accuracy
     */
    public static double avgError(DataOnMemory<DataInstance> dataWithMissing,
                                  DataOnMemory<DataInstance> trueData,
                                  DataOnMemory<DataInstance> imputedData) {

        List<Integer> continuousAttributesIndexes = dataWithMissing.getAttributes().getFullListOfAttributes().stream()
                .filter(Attribute::isContinuous)
                .map(Attribute::getIndex)
                .collect(Collectors.toList());

        List<Integer> discreteAttributesIndexes = dataWithMissing.getAttributes().getFullListOfAttributes().stream()
                .filter(Attribute::isDiscrete)
                .map(Attribute::getIndex)
                .collect(Collectors.toList());

        /* Estimate the maximum and minimum value of each continuous attribute (for error normalization) */
        DataUtils.defineAttributesMaxMinValues(trueData);

        double normalizedError = 0.0;

        /* Estimate the normalized error of continuous attributes */
        for(int j: continuousAttributesIndexes) {
            double attributeMissingValues = 0.0;
            double attributeError = 0.0;
            for(int i = 0; i < dataWithMissing.getNumberOfDataInstances(); i++) {
                if(Double.isNaN(dataWithMissing.getDataInstance(i).toArray()[j])) {
                    attributeMissingValues++;
                    double error = imputedData.getDataInstance(i).toArray()[j] - trueData.getDataInstance(i).toArray()[j];
                    attributeError += Math.pow(error, 2);
                }
            }
            attributeError = attributeError / attributeMissingValues;
            attributeError = Math.sqrt(attributeError);
            double attributeMax = ((RealStateSpace) trueData.getAttributes().getFullListOfAttributes().get(j).getStateSpaceType()).getMaxInterval();
            double attributeMin = ((RealStateSpace) trueData.getAttributes().getFullListOfAttributes().get(j).getStateSpaceType()).getMinInterval();
            double normalizedAttributeError = attributeError / (attributeMax - attributeMin);
            normalizedError += normalizedAttributeError;
        }

        /* Estimate the accuracy error of discrete attributes */
        for(int j: discreteAttributesIndexes) {
            double attributeMissingValues = 0.0;
            double attributeMissingValuesWrong = 0.0;
            for(int i = 0; i < dataWithMissing.getNumberOfDataInstances(); i++) {
                if (Double.isNaN(dataWithMissing.getDataInstance(i).toArray()[j])) {
                    attributeMissingValues++;
                    // Accuracy error
                    if(imputedData.getDataInstance(i).toArray()[j] != trueData.getDataInstance(i).toArray()[j])
                        attributeMissingValuesWrong++;
                }
            }
            double normalizedAttributeError = 0.0;
            if(attributeMissingValues > 0) // Avoid division by 0 in case there are 0 missing in that attribute
                normalizedAttributeError = attributeMissingValuesWrong / attributeMissingValues; // It is already normalized [0,1]
            normalizedError += normalizedAttributeError;
        }

        return normalizedError / trueData.getAttributes().getNumberOfAttributes();
    }
}
