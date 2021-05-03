package eu.amidst.extension.util.mi;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.variables.Variable;
import eu.amidst.extension.util.mi.util.DiscreteDataSet;

import java.util.HashMap;
import java.util.Map;

class DiscreteMI {

    public static double dd(double[][] data, Attribute x, Attribute y, boolean normalization) {
        if(!x.isDiscrete() || !y.isDiscrete())
            throw new IllegalArgumentException("Both X and Y have to be discrete");

        return dd(data, normalization);
    }

    public static double dd(double[][] data, Variable x, Variable y, boolean normalization) {
        if(!x.isDiscrete() || !y.isDiscrete())
            throw new IllegalArgumentException("Both X and Y have to be discrete");

        return dd(data, normalization);
    }

    private static double dd(double[] x, double[] y, boolean normalization) {

        if(x.length != y.length)
            throw new IllegalArgumentException("Both x and y must have the same number of items");

        double[][] data = new double[x.length][2];
        data[0] = x;
        data[1] = y;

        return dd(data, normalization);
    }

    private static double dd(double[][] data, boolean normalization) {

        /* Create the DiscreteDataSet, which will make it easier to estimate the MI */
        DiscreteDataSet xyCountsData = new DiscreteDataSet(data);

        Map<Double, Double> xFreqs= new HashMap<>();
        Map<Double, Double> yFreqs= new HashMap<>();
        Map<double[], Double> xyFreqs = new HashMap<>();

        /* Frequencies estimation */
        int N = data.length;
        for(double[] instance: xyCountsData) {
            double instanceCount = xyCountsData.getCounts(instance);
            double instanceFreq = instanceCount / N;

            double xFreq = 0;
            if(xFreqs.containsKey(instance[0]))
                xFreq = xFreqs.get(instance[0]);

            double yFreq = 0;
            if(yFreqs.containsKey(instance[1]))
                yFreq = yFreqs.get(instance[1]);

            xFreqs.put(instance[0], xFreq + instanceFreq);
            yFreqs.put(instance[1], yFreq + instanceFreq);
            xyFreqs.put(instance, instanceFreq);
        }

        /* Estimate entropies from frequencies */
        double Hx = 0;
        for(double instanceX: xFreqs.keySet()){
            double freq = xFreqs.get(instanceX);
            Hx -= freq * Math.log(freq);
        }

        double Hy = 0;
        for(double instanceY: yFreqs.keySet()){
            double freq = yFreqs.get(instanceY);
            Hy -= freq * Math.log(freq);
        }

        double Hxy = 0;
        for(double[] instance: xyFreqs.keySet()){
            double freq = xyFreqs.get(instance);
            Hxy -= freq * Math.log(freq);
        }

        double mi = Hx + Hy - Hxy;
        double normalizationFactor = Math.min(Hx, Hy);

        if(normalization)
            return mi / normalizationFactor;

        return mi;
    }
}
