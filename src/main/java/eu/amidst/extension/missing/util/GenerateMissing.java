package eu.amidst.extension.missing.util;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.extension.util.tuple.Tuple2;

import java.util.*;

public class GenerateMissing {

    /**
     *
     */
    public static DataOnMemory<DataInstance> randomlyHideValues(DataOnMemory<DataInstance> data,
                                                                double percentage,
                                                                long seed) {

        /* Copy dataset */
        DataOnMemory<DataInstance> copyData = data.deepCopy();

        /* Generate random pairs */
        Set<Tuple2<Integer, Integer>> pairs = new HashSet<>();
        do {
            pairs = generateRandomPairs(data, percentage, seed);
        } while (!checkAttributeWithAllMissing(pairs, data.getAttributes().getNumberOfAttributes(), data.getNumberOfDataInstances()));

        /* Assign Double.NaN to the selected pair value */
        for(Tuple2<Integer, Integer> pair: pairs) {
            Attribute attribute = copyData.getAttributes().getFullListOfAttributes().get(pair.getSecond());
            DataInstance instance = copyData.getDataInstance(pair.getFirst());
            instance.setValue(attribute, Double.NaN);
        }

        return copyData;
    }

    private static Set<Tuple2<Integer, Integer>> generateRandomPairs(DataOnMemory<DataInstance> data,
                                                                     double percentage,
                                                                     long seed) {

        /* Generate random pairs */
        Random randomGenerator = new Random(seed);

        int n_instances = data.getNumberOfDataInstances();
        int n_attributes = data.getAttributes().getNumberOfAttributes();

        int n_pairs = (int) (n_attributes * n_instances * percentage);
        Set<Tuple2<Integer, Integer>> pairs = new HashSet<>(n_pairs);

        while(pairs.size() < n_pairs) {
            int i = randomGenerator.nextInt(n_instances);
            int j = randomGenerator.nextInt(n_attributes);
            pairs.add(new Tuple2<>(i,j));
        }

        return pairs;
    }

    private static boolean checkInstanceWithAllMissing(Set<Tuple2<Integer, Integer>> pairs,
                                                       int n_attributes,
                                                       int n_instances) {

        /* Count missing per instance */
        Map<Integer, Integer> counts = new HashMap<>(n_instances);
        for(Tuple2<Integer, Integer> pair: pairs){

            if(!counts.containsKey(pair.getFirst()))
                counts.put(pair.getFirst(), 1);

            else {
                int n = counts.get(pair.getFirst()) + 1;

                /* If the number of missing is equal to the number of columns, return false */
                if(n == n_attributes)
                    return false;

                /* Else, update de count map*/
                counts.put(pair.getFirst(), n);
            }
        }

        /* There is no instance with all values missing */
        return true;
    }

    private static boolean checkAttributeWithAllMissing(Set<Tuple2<Integer, Integer>> pairs,
                                                        int n_attributes,
                                                        int n_instances) {
        /* Count missing per instance */
        Map<Integer, Integer> counts = new HashMap<>(n_attributes);
        for(Tuple2<Integer, Integer> pair: pairs){

            if(!counts.containsKey(pair.getSecond()))
                counts.put(pair.getSecond(), 1);

            else {
                int n = counts.get(pair.getSecond()) + 1;

                /* If the number of missing is equal to the number of columns, return false */
                if(n == n_instances)
                    return false;

                /* Else, update de count map*/
                counts.put(pair.getSecond(), n);
            }
        }

        /* There is no attribute with all values missing */
        return true;
    }
}
