package eu.amidst.extension.util.rfmetric;

import eu.amidst.core.models.DAG;

import java.util.List;

public class RobinsonFouldsMetric {

    public static double estimate(DAG firstDAG, DAG secondDAG) {

        /* Create the RF forests*/
        RFForest firstRFForest = RFForest.create(firstDAG);
        RFForest secondRFForest = RFForest.create(secondDAG);

        /* Estimate their partitions */
        List<RFPartition> firstPartitions = firstRFForest.partitions();
        List<RFPartition> secondPartitions = secondRFForest.partitions();


        /* All the partitions that appear on the first DAG but not in the second */
        int firstPartInSecondPart = 0;
        for(RFPartition first: firstPartitions)
            if(secondPartitions.contains(first))
                firstPartInSecondPart++;

        /* All the partitions that appear on the second DAG but not in the first */
        int secondPartInFirstPart = 0;
        for(RFPartition second: secondPartitions)
            if(firstPartitions.contains(second))
                secondPartInFirstPart++;

        /*
            Apply the formula:
            dRF = (|C1 - C1_2| + |C2 - C2_1|) / 2
         */
        return (Math.abs(firstPartitions.size() - firstPartInSecondPart) + Math.abs(secondPartitions.size() - secondPartInFirstPart))/2;
    }
}
