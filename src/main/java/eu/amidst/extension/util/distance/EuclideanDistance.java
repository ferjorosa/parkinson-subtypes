package eu.amidst.extension.util.distance;

public class EuclideanDistance implements DistanceFunction {

    @Override
    public double distance(double[] first, double[] second) {
        return Math.sqrt(squareDistance(first, second));
    }

    @Override
    public double squareDistance(double[] first, double[] second) {

        double dist = 0;
        double tmp;

        for(int i=0; i < first.length; i++) {
            tmp = first[i] - second[i];
            dist += tmp * tmp;
        }

        return dist;
    }

    @Override
    public double distance(double first, double second) {
        return Math.sqrt(squareDistance(first, second));
    }

    @Override
    public double squareDistance(double first, double second) {
        double tmp = first - second;
        return tmp * tmp;
    }
}
