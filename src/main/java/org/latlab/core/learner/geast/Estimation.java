package org.latlab.core.learner.geast;

import org.latlab.core.data.MixedDataSet;
import org.latlab.core.graph.AbstractNode;
import org.latlab.core.model.*;
import org.latlab.core.reasoner.*;
import org.latlab.core.util.DoubleComparator;

import java.util.*;

/**
 * An estimation is a combination of a model and a data set.
 * 
 * <p>
 * It makes a clone to the given model. After running this estimation, the
 * parameters in the clone is updated, and it can be retrieved by the
 * {@link #model()} method. The reference to the original model can be retrieved
 * by the {@link #origin} variable.
 * 
 * @author leonard
 * 
 */
public class Estimation implements IModelWithScore {

	/**
	 * Maps variables to the sufficient statistics need to estimation the
	 * parameters of the nodes corresponding to the variables. This class
	 * extends the base map so that we do not need to use its long name.
	 * 
	 * @author leonard
	 * 
	 */
	public static class VariableStatisticsMap
			extends
			MixedVariableMap<SufficientStatistics, MixedCliqueSufficientStatistics, SufficientStatistics> {

		public VariableStatisticsMap(int initialCapacity) {
			super(initialCapacity);
		}
	}

	/**
	 * Compares the BIC scores of two estimations in descending order (and thus
	 * the negate operator).
	 */
	public static final Comparator<Estimation> BIC_COMPARATOR =
			new Comparator<Estimation>() {
				public int compare(Estimation o1, Estimation o2) {
					return -DoubleComparator.compare(o1.BicScore(),
							o2.BicScore());
				}
			};

	/**
	 * Compares the loglikelihood of two estimations in descending order(and
	 * thus the negate operator).
	 */
	public static final Comparator<Estimation> LOGLIKELIHOOD_COMPARATOR =
			new Comparator<Estimation>() {
				public int compare(Estimation o1, Estimation o2) {
					return -DoubleComparator.compare(o1.loglikelihood(),
							o2.loglikelihood());
				}
			};

	public static final EstimationFactory FACTORY =
			new EstimationFactory(new EstimationFactory.Prototype() {
				public Estimation create(Gltm model, Gltm clone,
						MixedDataSet data, DataPropagation propagation,
						double smoothing) {
					return new Estimation(model, clone, data, propagation,
							smoothing);
				}

				public int partitions() {
					return 1;
				}
			});

	protected final Gltm model;

	/**
	 * Holds the origin of this model. The current model is estimated from the
	 * origin model and share the same structure of it.
	 */
	private final Gltm origin;
	protected final MixedDataSet data;

	protected final double smoothing;

	protected double loglikelihood = -Double.MAX_VALUE;
	private double previousLoglikelihood = Double.NaN;
	private double bicScore = Double.NaN;

	protected DataPropagation propagation;

	protected int messagesPassed = 0;

	// TODO: consider not to clone the model, but generate a new model when it
	// is needed.
	/**
	 * 
	 * @param model
	 * @param data
	 * @param smoothing
	 */
	protected Estimation(Gltm origin, Gltm model, MixedDataSet data,
			DataPropagation propagation, double smoothing) {
		this.origin = origin;
		this.model = model;
		this.data = data;
		this.smoothing = smoothing;

		this.propagation = propagation;
	}

	/**
	 * Holds an instance of sufficient statistics for each clique. This list has
	 * the same order as the clique tree nodes in the clique tree.
	 */
	protected List<SufficientStatistics> sufficientStatistics;

	/**
	 * Returns the model with its estimated parameters.
	 * 
	 * @return model with estimated parameters
	 */
	public Gltm model() {
		return model;
	}

	public Gltm origin() {
		return origin;
	}

	// public boolean sharesSameOrigin(Estimation other) {
	// return origin == other.origin;
	// }

	public MixedDataSet data() {
		return data;
	}

	public void savePreviousLoglikelihood() {
		previousLoglikelihood = loglikelihood;
	}

	public double loglikelihood() {
		return loglikelihood;
	}

	/**
	 * Returns the improvement over the last saved loglikelihood. If the
	 * likelihood has not been saved, a very large value is returned.
	 * 
	 * @return improvement over the last saved loglikelihood
	 */
	public double improvement() {
		return Double.isNaN(previousLoglikelihood) ? Double.MAX_VALUE
				: loglikelihood - previousLoglikelihood;
	}

	public void setInvalidLoglikehood() {
		loglikelihood = Double.NaN;
		bicScore = Double.NaN;
	}

	/**
	 * Returns BIC score by deriving from the previous computed loglikelihood.
	 * 
	 * @return BIC score
	 */
	public double BicScore() {
		if (Double.isNaN(bicScore))
			bicScore =
					loglikelihood - model.computeDimension()
							* Math.log(data.totalWeight()) / 2.0;

		return bicScore;
	}

	private void reset() {
		loglikelihood = 0;
		bicScore = Double.NaN;
		if (sufficientStatistics == null) {
			initializeSufficientStatistics();
		} else {
			resetSufficientStatistics();
		}
	}

	private void initializeSufficientStatistics() {
		NaturalCliqueTree tree = propagation.cliqueTreeStructure();
		sufficientStatistics =
				new ArrayList<SufficientStatistics>(tree.getNodes().size());

		for (AbstractNode node : tree.getNodes()) {
			SufficientStatistics statistics =
					((CliqueTreeNode) node).accept(new SufficientStatistics.Constructor(
							smoothing));
			sufficientStatistics.add(statistics);
		}
	}

	private void resetSufficientStatistics() {
		for (SufficientStatistics statistics : sufficientStatistics) {
			statistics.reset();
		}
	}

	public VariableStatisticsMap collectSufficientStatistics() {
		reset();
		computeSufficientStatistics();
		return constructVariableToStatisticsMap();
	}

	protected void computeSufficientStatistics() {
		messagesPassed = 0;

		for (int i = 0; i < data.size(); i++) {
			NaturalCliqueTreePropagation ctp = propagation.compute(i);
			messagesPassed += ctp.messagesPassed();

			Iterator<SufficientStatistics> statisticsIterator =
					sufficientStatistics.iterator();

			for (AbstractNode node : ctp.cliqueTree().getNodes()) {
				final SufficientStatistics statistics =
						statisticsIterator.next();
				statistics.add(((CliqueTreeNode) node).potential(),
						data.get(i).weight());
			}

			double weight = data.get(i).weight();
			loglikelihood += ctp.loglikelihood() * weight;
		}
	}

	/**
	 * The sufficient statistics for a variable not in focus is not put into the
	 * map.
	 * 
	 * @return
	 */
	private VariableStatisticsMap constructVariableToStatisticsMap() {
		// use the cliques from one single tree as links between two maps
		final NaturalCliqueTree tree = propagation.cliqueTreeStructure();

		// build a map from clique to sufficient statistics
		final Map<AbstractNode, SufficientStatistics> cliqueMap =
				new HashMap<AbstractNode, SufficientStatistics>(
						sufficientStatistics.size());
		Iterator<AbstractNode> nodeIterator = tree.getNodes().iterator();
		for (SufficientStatistics statistics : sufficientStatistics) {
			cliqueMap.put(nodeIterator.next(), statistics);
		}

		// build a map from variable to sufficient statistics
		final VariableStatisticsMap variableMap =
				new VariableStatisticsMap(sufficientStatistics.size());

		for (AbstractNode node : model.getNodes()) {
			((BeliefNode) node).accept(new BeliefNode.Visitor<Void>() {

				public Void visit(DiscreteBeliefNode node) {
					Clique clique = tree.getClique(node.getVariable());
					if (!clique.focus())
						return null;

					variableMap.put(node.getVariable(), cliqueMap.get(clique));
					return null;
				}

				public Void visit(ContinuousBeliefNode node) {
					Clique clique = tree.getClique(node.getVariable());
					if (!clique.focus())
						return null;

					variableMap.put(
							node.getVariable(),
							(MixedCliqueSufficientStatistics) cliqueMap.get(clique));

					return null;
				}

			});
		}

		return variableMap;
	}

	/**
	 * Indicates the completion of this estimation and releases the memory of
	 * the objects used for making estimation. It cannot be used for doing any
	 * other estimation on this instance.
	 */
	public void complete() {
		propagation = null;
	}

	public double computeLogLikelihood() {
		loglikelihood = 0;

		for (int i = 0; i < data.size(); i++) {
			NaturalCliqueTreePropagation ctp = propagation.compute(i);
			double weight = data.get(i).weight();
			loglikelihood += ctp.loglikelihood() * weight;
		}

		return loglikelihood;
	}

	/**
	 * Computes the loglikelihood, ignore the impossible evidence case. Returns
	 * the number of valid data cases.
	 * 
	 * @return number of valid data cases
	 */
	public double computeLoglikelihoodWithIgnore() {
		loglikelihood = 0;
		double valid = 0;

		for (int i = 0; i < data.size(); i++) {
			try {
				NaturalCliqueTreePropagation ctp = propagation.compute(i);
				double weight = data.get(i).weight();
				loglikelihood += ctp.loglikelihood() * weight;
				valid += weight;
			} catch (ImpossibleEvidenceException e) {
				// ignore
			}
		}

		return valid;
	}

	public String toString() {
		return String.format(
				"Estimation: model=[%s] loglikelihood=[%f], BIC=[%f]",
				model.getName(), loglikelihood, bicScore);
	}

	public int messageCount() {
		return messagesPassed;
	}
}
