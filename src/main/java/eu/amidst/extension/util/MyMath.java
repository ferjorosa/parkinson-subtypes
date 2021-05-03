package eu.amidst.extension.util;

import java.util.List;

public class MyMath {

    public static double mean(double[] x) {
        return sum(x) / x.length;
    }

    public static double mean(List<Double> x) {
        return sum(x) / x.size();
    }

    public static double stDev(double[] x) {
        return stDev(x, mean(x));
    }

    public static double stDev(double[] x, double mean) {

        double sum = 0;

        for (int j = 0; j < x.length; j++) {
            // put the calculation right in there
            sum = sum + ((x[j] - mean) * (x[j] - mean));
        }
        double squaredDiffMean = (sum) / (x.length);

        return (Math.sqrt(squaredDiffMean));
    }

    public static double stDev(List<Double> x, double mean) {

        double sum = 0;

        for (int j = 0; j < x.size(); j++) {
            // put the calculation right in there
            sum = sum + ((x.get(j) - mean) * (x.get(j) - mean));
        }
        double squaredDiffMean = (sum) / (x.size());

        return (Math.sqrt(squaredDiffMean));
    }

    public static double sum(double[] x) {
        double sum = 0.0;

        for (double n : x) {
            sum += n;
        }

        return sum;
    }

    public static double sum(List<Double> x) {
        double sum = 0.0;

        for (double n : x) {
            sum += n;
        }

        return sum;
    }

    public static long factorial(int number) {
        long result = 1;

        for (int factor = 2; factor <= number; factor++) {
            result *= factor;
        }

        return result;
    }

    public static double log(double a) {
        if(a == 0)
            return 0;
        return Math.log(a);
    }
}
