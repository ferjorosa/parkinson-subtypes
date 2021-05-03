package org.latlab.core.learner.geast;

import org.latlab.core.model.CGParameter;
import org.latlab.core.model.CGPotential;
import org.latlab.core.util.DiscreteVariable;
import org.latlab.core.util.Function;
import org.latlab.core.util.JointContinuousVariable;
import org.latlab.core.util.Potential;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;

public class MixedCliqueSufficientStatistics implements SufficientStatistics {
    private final NormalSufficientStatistics[] statistics;
    final private double smoothing;

    public MixedCliqueSufficientStatistics(
        int headVariables, int parentStates, double smoothing) {
        statistics = new NormalSufficientStatistics[parentStates];
        for (int i = 0; i < statistics.length; i++)
            statistics[i] = new NormalSufficientStatistics(headVariables);

        this.smoothing = smoothing;
    }

    public void reset() {
        for (NormalSufficientStatistics s : statistics) {
            s.reset();
        }
    }

    public void add(Potential potential, double weight) {
        add((CGPotential) potential, weight);
    }

    public void add(CGPotential potential, double weight) {
        for (int i = 0; i < potential.size(); i++) {
            statistics[i].add(potential.get(i), weight);
        }
    }

    public CGPotential computePotential(
        JointContinuousVariable head, DiscreteVariable parent) {
        CGParameter[] parameters = new CGParameter[statistics.length];
        for (int i = 0; i < parameters.length; i++) {
            parameters[i] =
                new CGParameter(1, statistics[i].computeMean(), statistics[i]
                    .computeCovariance());
        }

        return new CGPotential(head, parent, parameters);
    }

    public Function computePotential(
        DiscreteVariable variable, Collection<DiscreteVariable> parents) {
        assert parents.size() == 0;
        assert statistics.length == variable.getCardinality();

        Function potential =
            Function.createFunction(Collections.singletonList(variable));
        double[] cells = potential.getCells();
        for (int i = 0; i < statistics.length; i++) {
            cells[i] = statistics[i].p + smoothing;
        }

        // to handle the case when all entries are zeros
        potential.normalize(variable);

        return potential;
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "\n" + Arrays.toString(statistics);
    }

    @Override
    public boolean equals(Object o) {
        if (o == this)
            return true;
        else if (o == null)
            return false;
        else if (o instanceof MixedCliqueSufficientStatistics) {
            return Arrays.deepEquals(
                statistics, ((MixedCliqueSufficientStatistics) o).statistics);
        } else
            return false;
    }
    
    @Override
    public int hashCode() {
        return Arrays.deepHashCode(statistics);
    }
    
    /**
     * Exposed for testing only.
     */
    NormalSufficientStatistics[] statistics() {
        return statistics;
    }
}
