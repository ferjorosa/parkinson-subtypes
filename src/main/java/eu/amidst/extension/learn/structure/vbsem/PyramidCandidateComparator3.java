package eu.amidst.extension.learn.structure.vbsem;

import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.extension.util.tuple.Tuple3;

import java.util.Comparator;

public class PyramidCandidateComparator3 implements Comparator<Tuple3<BayesianNetwork, Double, Boolean>> {

    @Override
    public int compare(Tuple3<BayesianNetwork, Double, Boolean> o1, Tuple3<BayesianNetwork, Double, Boolean> o2) {
        if(o1.getSecond() > o2.getSecond())
            return -1;
        else if(o1.getSecond() < o2.getSecond())
            return 1;
        else
            return 0;
    }
}
