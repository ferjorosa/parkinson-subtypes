package org.latlab.core.learner.geast;

import cern.colt.function.DoubleDoubleFunction;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.math.Functions;
import org.latlab.core.model.CGParameter;
import org.latlab.core.util.MixtureOfGaussianStructure;

/**
 * Sufficient statistics for a mixture of normal distribution. {@code p} holds
 * T1, {@code A} T2, {@code C} T3.
 * 
 * @author leonard
 * 
 */
public class NormalSufficientStatistics extends MixtureOfGaussianStructure {

    public NormalSufficientStatistics(int variables) {
        super(variables, 0);
    }

	public void add(CGParameter parameter, double weight) {
        add(parameter.A, parameter.C, weight * parameter.p);
    }

    /**
     * This assumes that the covariance matrix is zero.
     * 
     * @param observation
     * @param weight
     *            the number of occurrences of this observation, which is
     *            usually computed by multiplying probability of this
     *            observation with data weight
     */
    public void add(DoubleMatrix1D observation, double weight) {
        DoubleDoubleFunction plusMult = Functions.plusMult(weight);
        // p += P(z|d)
        p += weight;

        // A += P(z|d) * y
        A.assign(observation, plusMult);

        // C += P(z|d) * (y x y')
        DoubleMatrix2D product =
            Algebra.DEFAULT.multOuter(observation, observation, null);
        C.assign(product, plusMult);
    }
    
    /**
     * @param mean mu_i = E[x_i]
     * @param covariance sigma_ij = E[x_i x_j]
     * @param weight
     */
    public void add(DoubleMatrix1D mean, DoubleMatrix2D covariance, double weight) {
    	add(mean, weight);
    	C.assign(covariance, Functions.plusMult(weight));
    }

    /**
     * Resets this statistics and set the entries to {@code 0}.
     */
    public void reset() {
        this.p = 0;
        A.assign(0);
        C.assign(0);
    }

    /**
     * Computes and returns the maximum likelihood estimates of the mean vector.
     * 
     * @return maximum likelihood estimates of the mean vector
     */
    public DoubleMatrix1D computeMean() {
        // If the p is zero, return a zero vector as mean to prevent
        // divided by zero. p equals to zero implies that this component is
        // impossible to occur, so the parameters do not matter anyway.
        if (p == 0) {
            return new DenseDoubleMatrix1D(A.size());
        }

        // mean = T2 / T1
        DoubleMatrix1D mean = A.copy();
        mean.assign(Functions.div(p));

        return mean;
    }

    /**
     * Computes and returns the maximum likelihood estimates of the covariance
     * matrix.
     * 
     * @return maximum likelihood estimates of the covariance matrix
     */
    public DoubleMatrix2D computeCovariance() {
        // if the p is zero, return a zero matrix as covariance to prevent
        // divided by zero. p equals to zero implies that this component is
        // impossible to occur, so the parameters do not matter anyway.
        if (p == 0) {
            return new DenseDoubleMatrix2D(A.size(), A.size());
        }

        // using a divider of p^2 may lead to a divided by zero precision error,
        // so we have to use a slightly less efficient approach
//        // covariance = [T3 - (T2 x T2')/T1]/T1 = [(T2 x T2')/(-T1*T1) + T3/T1
//        DoubleMatrix2D covariance = Algebra.DEFAULT.multOuter(A, A, null);
//        covariance.assign(Functions.div(-p * p));
//        covariance.assign(C, Functions.plusMult(1 / p));

        // covariance = [T3 - (T2 x T2')/T1]/T1 = [(T2 x T2')/-T1 + T3]/T1
        DoubleMatrix2D covariance = Algebra.DEFAULT.multOuter(A, A, null);
        covariance.assign(Functions.div(-p));
        covariance.assign(C, Functions.plus);
        covariance.assign(Functions.div(p));
        
        return covariance;
    }
}
