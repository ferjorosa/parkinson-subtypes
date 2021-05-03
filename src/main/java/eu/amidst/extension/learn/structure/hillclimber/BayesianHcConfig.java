package eu.amidst.extension.learn.structure.hillclimber;

/** The objective of this class is to store the parameters necessary for VB that will be used in the learning process
 * inside HC operators.
 */
public class BayesianHcConfig {

    private long seed;

    private double threshold;

    private int maxIter;

    public BayesianHcConfig() {
        this.seed = 0;
        this.threshold = 0.01;
        this.maxIter = 1; // Only 1 iteration is necessary when no latent variables are present
    }

    public BayesianHcConfig(long seed, double threshold, int maxIter) {
        this.seed = seed;
        this.threshold = threshold;
        this.maxIter = maxIter;
    }

    public long getSeed() {
        return seed;
    }

    public double getThreshold() {
        return threshold;
    }

    public int getMaxIter() {
        return maxIter;
    }
}
