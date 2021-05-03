package eu.amidst.extension.learn.parameter;

import eu.amidst.extension.util.tuple.Tuple2;

import java.util.Comparator;

public class PyramidCandidateComparator<T> implements Comparator<Tuple2<T, Double>> {

    @Override
    public int compare(Tuple2<T, Double> o1, Tuple2<T, Double> o2) {
        if(o1.getSecond() > o2.getSecond())
            return -1;
        else if(o1.getSecond() < o2.getSecond())
            return 1;
        else
            return 0;
    }
}
