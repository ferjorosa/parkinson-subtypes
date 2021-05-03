package eu.amidst.extension.util.distance;

public interface DistanceFunction {

    /**
     * Calculates the distances between two vectors.
     * @param first first vector.
     * @param second second vector.
     * @return their distance value.
     */
    double distance(double[] first, double[] second);

    /**
     * Calculates the square distance between two vectors. Also known as the reduced distance, it is often used to speed
     * computations, such as in the nearest neighbour algorithms.
     * @param first first vector.
     * @param second second vector.
     * @return their distance value.
     */
    double squareDistance(double[] first, double[] second);

    /**
     * Calculates the distance between two values
     * @param first first value
     * @param second second value
     * @return their distance value
     */
    double distance(double first, double second);

    /**
     * Calculates the distance between two values
     * @param first first value
     * @param second second value
     * @return their distance value
     */
    double squareDistance(double first, double second);
}
