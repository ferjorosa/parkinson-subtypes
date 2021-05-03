package org.latlab.core.learner.geast;

import org.latlab.core.util.SingularContinuousVariable;

import java.io.PrintWriter;
import java.util.Collection;

public class ConstantCovarianceConstrainer extends CovarianceConstrainer {

    public final static double DEFAULT_LOWER_BOUND = 0.01;
    public final static double DEFAULT_UPPER_BOUND = Double.POSITIVE_INFINITY;

    public final double lower;
    public final double upper;

    public ConstantCovarianceConstrainer() {
        this(DEFAULT_LOWER_BOUND, Double.POSITIVE_INFINITY);
    }

    @Override
    protected double getLowerBound(
        Collection<SingularContinuousVariable> variables) {
        return lower;
    }

    protected double getUpperBound(
        Collection<SingularContinuousVariable> variables) {
        return upper;
    }

    /**
     * 
     * @param lower
     *            lower bound of the eigenvalue of the covariance matrix
     * @param upper
     *            upper bound of the eigenvalue of the covariance matrix
     */
    public ConstantCovarianceConstrainer(double lower, double upper) {
        assert lower <= upper;

        this.lower = lower;
        this.upper = upper;
    }

    @Override
    public void writeXml(PrintWriter writer) {
        writer.format("<covarianceConstraints type='constant' "
            + "eigenvalueLower='%s' eigenvalueUpper='%s' />", lower, upper);
        writer.println();
    }
}
