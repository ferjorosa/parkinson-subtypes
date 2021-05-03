package org.latlab.core.learner.geast;

import cern.colt.matrix.DoubleMatrix1D;
import org.latlab.core.data.MixedDataSet;
import org.latlab.core.util.SingularContinuousVariable;

import java.io.PrintWriter;
import java.util.Collection;

/**
 * Uses the variance
 * 
 * @author leonard
 * 
 */
public class VariableCovarianceConstrainer extends CovarianceConstrainer {

    public static final double DEFAULT_MULTIPLIER = 20;
    private final MixedDataSet data;
    public final double multiplier;
    public final boolean hasUpperBound;

    public VariableCovarianceConstrainer(
        MixedDataSet data, double multiplier, boolean hasUpperBound) {
        this.data = data;
        this.multiplier = multiplier;
        this.hasUpperBound = hasUpperBound;
    }

    @Override
    protected double getLowerBound(
        Collection<SingularContinuousVariable> variables) {
        double min = Double.MAX_VALUE;

        DoubleMatrix1D variance = data.variance();

        for (SingularContinuousVariable variable : variables) {
            int index = data.indexOf(variable);
            min = Math.min(min, variance.getQuick(index));
        }

        return min / multiplier;
    }

    @Override
    protected double getUpperBound(
        Collection<SingularContinuousVariable> variables) {
        if (!hasUpperBound)
            return Double.POSITIVE_INFINITY;

        double max = -Double.MAX_VALUE;

        DoubleMatrix1D variance = data.variance();

        for (SingularContinuousVariable variable : variables) {
            int index = data.indexOf(variable);
            max = Math.max(max, variance.getQuick(index));
        }

        return max * multiplier;
    }

    @Override
    public void writeXml(PrintWriter writer) {
        writer.format(
            "<covarianceConstraints type='variable' "
                + "multiplier='%s' hasUpperBound='%s' />", multiplier,
            hasUpperBound);
        writer.println();
    }
}
