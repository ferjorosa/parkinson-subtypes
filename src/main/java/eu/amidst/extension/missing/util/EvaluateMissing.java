package eu.amidst.extension.missing.util;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;

/**
 *
 */
public class EvaluateMissing {

    public static double accuracy(List<Double> v1, List<Double> v2) {
        if(v1.size() != v2.size())
            throw new IllegalArgumentException("Both vectors must have the same length");

        double total_right = 0;
        for(int i = 0; i < v1.size(); i++)
            if(v1.get(i).equals(v2.get(i)))
                total_right++;

        return total_right / v1.size();
    }

    public static double mse(List<Double> v1, List<Double> v2) {
        if(v1.size() != v2.size())
            throw new IllegalArgumentException("Both vectors must have the same length");

        double mse = 0;
        double error = 0;
        for(int i = 0; i < v1.size(); i++) {
            error = v1.get(i) - v2.get(i);
            mse += Math.pow(error, 2)/v1.size();
        }

        return mse;
    }

    public static List<Double> generateVectorForEvaluation(DataOnMemory<DataInstance> data,
                                                       LinkedHashMap<Integer, List<Attribute>> missingValueLocations) {

        List<Double> values = new ArrayList<>();
        for(int instanceIndex: missingValueLocations.keySet()) {
            for(Attribute attribute: missingValueLocations.get(instanceIndex))
                values.add(data.getDataInstance(instanceIndex).getValue(attribute));
        }

        return values;
    }
}
