package org.latlab.core.util;

import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.linalg.CholeskyDecomposition;
import cern.jet.math.Functions;

import java.util.Random;

public class Normal {
	
	private final static double PI_TIMES_2 = 2* Math.PI;
	private final static double LOG_2_PI = Math.log(2 * Math.PI);

    /**
     * Returns the probability density function at the specified {@code value}
     * of a normal distribution N({@code mean}, {@code variance}).
     * 
     * @param mean
     *            mean of normal distribution
     * @param variance
     *            variance of normal distribution
     * @param value
     *            value to evaluate
     * @return probability dense function at {@code value}
     */
    public static double pdf(double mean, double variance, double value) {
        if (variance == 0) {
            return value == mean ? 1 : 0;
        }

        // 1/sqrt(2 * pi * variance) exp { -1/2 * [(value - mean)^2/variance] }
        double diff = (value - mean);
        double exp = Math.exp((diff * diff) / (-2 * variance));
        return exp / Math.sqrt(PI_TIMES_2 * variance);
    }
    
    public static double logPdf(double mean, double variance, double value) {
        if (variance == 0) {
            return value == mean ? 0 : Double.NEGATIVE_INFINITY;
        }

        // -1/2 * [log(2*pi) + log(variance)] + 
        //		{ -1/2 * [(value - mean)^2/variance] }
        double diff = (value - mean);
        double log = (diff * diff) / (-2 * variance);
        return log - (LOG_2_PI + Math.log(variance))/2;
    }

    private final DoubleMatrix1D mean;

    /**
     * The lower triangular matrix of Cholesky decomposition of covariance
     * matrix.
     */
    private final DoubleMatrix2D L;

    private final DoubleMatrix1D standardDeviation;

    //private final cern.jet.random.Normal standard = new cern.jet.random.Normal(0, 1, RandomEngine.makeDefault());
    private final Random standard = new Random(0);


    public Normal(DoubleMatrix1D mean, DoubleMatrix2D covariance) {
        this.mean = mean.copy();

        standardDeviation = DoubleFactory2D.dense.diagonal(covariance);
        standardDeviation.assign(Functions.sqrt);

        CholeskyDecomposition decomposition =
            new CholeskyDecomposition(covariance);
        if (decomposition.isSymmetricPositiveDefinite()) {
            L = new CholeskyDecomposition(covariance).getL();
        } else {
            L = DoubleFactory2D.dense.diagonal(standardDeviation);
        }

    }

    public Normal(DoubleMatrix1D mean, DoubleMatrix1D variance) {
        this.mean = mean.copy();

        standardDeviation = variance.copy();
        standardDeviation.assign(Functions.sqrt);

        // The covariance matrix is a diagonal matrix with variances.
        // The L of a diagonal matrix is a diagonal matrix with the square root
        // of the diagonal entries.
        L = DoubleFactory2D.dense.diagonal(standardDeviation);
    }

    /**
     * Constructs a normal distribution without specifying mean and covariance.
     * It allows only calls on method {@code generateWith} with arguments of
     * mean and covariance.
     */
    public Normal() {
        mean = null;
        L = null;
        standardDeviation = null;
    }

    /**
     * The steps for generating a random vector from this distribution:
     * <ol>
     * <li>Compute Cholesky decomposition L of covariance matrix.
     * <li>Generate a vector Z from N(0,I).
     * <li>X = mean + LZ
     * </ol>
     * 
     * @return vector generated from this distribution
     */
    public synchronized DoubleMatrix1D generate() {
        return generateWithL(mean, L);
    }

    public synchronized DoubleMatrix1D generate(int[] subset) {
        return generateWithL(mean.viewSelection(subset), L.viewSelection(
            subset, subset));
    }

    public DoubleMatrix1D generateWith(
            DoubleMatrix1D mean, DoubleMatrix2D covariance) {
        DoubleMatrix2D l = new CholeskyDecomposition(covariance).getL();
        return generateWithL(mean, l);

    }

    private DoubleMatrix1D generateWithL(DoubleMatrix1D m, DoubleMatrix2D l) {
        // X = mean + Lz
        DoubleMatrix1D vector =
            Algebra.DEFAULT.mult(l, generateStandardVector(l.columns()));
        return vector.assign(m, Functions.plus);
    }

    /**
     * Generates a vector from the distribution, assuming that each variable is
     * independent from each other.
     * 
     * @param subset
     *            subset of the variable
     * @return vector generated from the distribution
     */
    public synchronized DoubleMatrix1D generateIndependentVector(int[] subset) {
        DoubleMatrix1D vector = new DenseDoubleMatrix1D(subset.length);
        for (int i = 0; i < subset.length; i++) {
            int j = subset[i];
            double value =
                nextDouble(mean.getQuick(j), standardDeviation.getQuick(j));
            vector.setQuick(i, value);
        }

        return vector;
    }

    /**
     * Generates a vector from a standard normal distribution N(0,I)
     * 
     * @param length
     *            length of the generated vector
     * @return vector generated
     */
    private DoubleMatrix1D generateStandardVector(int length) {
        DoubleMatrix1D vector = new DenseDoubleMatrix1D(length);
        for (int i = 0; i < vector.size(); i++) {
            //vector.setQuick(i, standard.nextDouble());
            vector.setQuick(i, standard.nextGaussian());
        }

        return vector;
    }

    private double nextDouble(double mean, double standardDeviation) {
        //return mean + standard.nextDouble() * standardDeviation;
        return mean + standard.nextGaussian() * standardDeviation;
    }
}
