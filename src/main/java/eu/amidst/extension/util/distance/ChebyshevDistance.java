package eu.amidst.extension.util.distance;

public class ChebyshevDistance implements DistanceFunction {

    @Override
    public double distance(double[] first, double[] second) {
        return squareDistance(first, second);
    }

    @Override
    public double squareDistance(double[] first, double[] second) {

        double dist = 0;
        double tmp;

        for(int i=0; i < first.length; i++) {
            tmp = Math.abs(first[i] - second[i]);
            if(tmp > dist)
                dist = tmp;
        }

        return dist;
    }

    @Override
    public double distance(double first, double second) {
        return squareDistance(first, second);
    }

    @Override
    public double squareDistance(double first, double second) {
        return Math.abs(first - second);
    }
}
