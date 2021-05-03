package eu.amidst.extension.missing.util;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.stream.Collectors;

public class LocateMissing {

    public static LinkedHashMap<Integer, List<Attribute>> locateMissingValues(DataOnMemory<DataInstance> data) {

        /* Attributes in each data instance with a missing value */
        LinkedHashMap<Integer, List<Attribute>> missingValues = new LinkedHashMap<>();

        for(int i = 0; i < data.getNumberOfDataInstances(); i++) {
            DataInstance instance = data.getDataInstance(i);
            double[] values = instance.toArray();
            List<Integer> missingIndexes = new ArrayList<>();
            for (int j = 0; j < values.length; j++) {
                if (Double.isNaN(values[j]))
                    missingIndexes.add(j);
                if(!missingIndexes.isEmpty()) {
                    List<Attribute> attributesWithMissing = missingIndexes.stream()
                            .map(data.getAttributes().getFullListOfAttributes()::get)
                            .collect(Collectors.toList());
                    missingValues.put(i, attributesWithMissing);
                }
            }
        }

        return missingValues;
    }
}
