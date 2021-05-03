package org.latlab.core.learner.geast;

import org.latlab.core.reasoner.CliqueTreeNode;
import org.latlab.core.reasoner.DiscreteClique;
import org.latlab.core.reasoner.MixedClique;
import org.latlab.core.reasoner.Separator;
import org.latlab.core.util.DiscreteVariable;
import org.latlab.core.util.Function;
import org.latlab.core.util.Potential;

import java.util.Collection;

/**
 * Holds the sufficient statistics necessary for ML estimation of the parameters
 * of a clique in the clique tree corresponding to a Bayesian network.
 * 
 * @author leonard
 * 
 */
public interface SufficientStatistics {
    /**
     * Constructs a sufficient statistics instance according to different types
     * of clique tree node.
     * 
     * @author leonard
     * 
     */
    public static class Constructor
        implements CliqueTreeNode.Visitor<SufficientStatistics> {
        private final double smoothing;

        public Constructor(double smoothing) {
            this.smoothing = smoothing;
        }

        public SufficientStatistics visit(Separator separator) {
            return NIL;
        }

        public SufficientStatistics visit(DiscreteClique clique) {
            return clique.focus() == true
                ? new DiscreteCliqueSufficientStatistics(clique
                    .discreteVariables(), smoothing) : NIL;
        }

        public SufficientStatistics visit(MixedClique clique) {
            return clique.focus() == true
                ? new MixedCliqueSufficientStatistics(clique.jointVariable()
                    .variables().size(), clique.discreteVariable()
                    .getCardinality(), smoothing) : NIL;
        }
    }

    /**
     * An nil instance does not hold any statistics. It does nothing with its
     * methods.
     */
    public final static SufficientStatistics NIL = new SufficientStatistics() {
        public void reset() {}

        public void add(Potential potential, double weight) {}

        public Function computePotential(
            DiscreteVariable variable, Collection<DiscreteVariable> parents) {
            return null;
        }
    };

    /**
     * Resets this statistics, usually by setting all entries to zero.
     */
    public void reset();

    /**
     * Extracts the statistics from a given clique, and adds them to this
     * instance.
     * 
     * <p>
     * The potential should hold the computed distribution for a data case. The
     * data case can be adjusted by the given {@code weight}.
     * 
     * @param potential
     *            holds the distribution of the variables
     * @param weight
     *            weight of a data case
     */
    public void add(Potential potential, double weight);

    /**
     * Computes the maximum likelihood estimate of a discrete distribution based
     * on the collected sufficient statistics. The returned potential is a
     * (conditional) probability distribution of the given {@code variable},
     * with other variables as parent variables if there are any.
     * 
     * <p>
     * It assumes the sufficient statistics has the given {@code variable}.
     * 
     * <p>
     * Since it shouldn't have many parents in the latent tree model, it uses a
     * {@code Collection} to hold the parent variables. If there are many, it
     * should consider using a {@code Set} to hold them.
     * 
     * @param variable
     *            variable of the returned distribution
     * @return (conditional) probability distribution of the {@code variable}
     */
    public Function computePotential(
            DiscreteVariable variable, Collection<DiscreteVariable> parents);
}
