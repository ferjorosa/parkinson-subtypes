package org.latlab.core.learner.geast;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import org.latlab.core.data.MixedDataSet;
import org.latlab.core.graph.AbstractNode;
import org.latlab.core.learner.geast.Estimation.VariableStatisticsMap;
import org.latlab.core.model.*;
import org.latlab.core.reasoner.ImpossibleEvidenceException;

import java.io.PrintWriter;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;

/**
 * Calls {@link #setMultithreading(int, ExecutorService)} to use the
 * multithreading version.
 * 
 * @author leonard
 * 
 */
public abstract class EmFramework {

	protected final MixedDataSet data;
	private final ParameterGenerator generator;
	private CovarianceConstrainer constrainer;
	private EstimationFactory estimationFactory = Estimation.FACTORY;

	static class Instance {
		public Instance(Gltm model, Gltm origin, Focus focus) {
			this.model = model;
			this.origin = origin;
			this.focus = focus;
		}

		/**
		 * Should only be called after the number of candidates are reduced to
		 * only one.
		 * 
		 * @return best candidate
		 */
		public Estimation best() {
			assert size == 1;
			return candidates[0];
		}

		public final Gltm model;
		public final Gltm origin;
		public final Focus focus;

		public int stepsRun = 0;
		public Estimation[] candidates = null;
		public int size = 0;

		public boolean extremeDetected = false;

		// it overrides the settings of EM and has to generate a new parameters,
		// useful when the original parameters lead to NaN estimation
		public boolean forceGenerateParameters = false;
	}

	protected EmParameters parameters = new EmParameters();
	private long seed;

	public EmFramework(MixedDataSet data) {
		this.data = data;
		this.generator = new ParameterGenerator(data);
		this.constrainer = new ConstantCovarianceConstrainer();
	}

	/**
	 * Uses the given parameters and returns the old parameters.
	 * 
	 * @param parameters
	 *            parameters for EM
	 * @return old parameters
	 */
	public EmParameters use(EmParameters parameters) {
		EmParameters old = this.parameters;
		this.parameters = parameters;
		return old;
	}

	public CovarianceConstrainer useCovarianceConstrainer(
			CovarianceConstrainer constrainer) {
		CovarianceConstrainer old = this.constrainer;
		this.constrainer = constrainer;
		return old;
	}

	public Estimation estimate(IModelWithScore current) {
		return estimate(new Instance(current.model(), current.origin(), null));
	}

	/**
	 * Runs an EM on a current estimation. The returned estimation has the same
	 * origin as that of the current one.
	 * 
	 * @param current
	 * @param focus
	 *            the focus of the estimation, which may help speed up by some
	 *            derived class, where {@code null} means focus on the whole
	 *            model
	 * @return
	 */
	public Estimation estimate(IModelWithScore current, Focus focus) {
		return estimate(new Instance(current.model(), current.origin(), focus));
	}

	public Estimation estimate(Gltm model) {
		return estimate(new Instance(model, model, null));
	}

	public Estimation estimate(Gltm model, Focus focus) {
		// TODO LP: may have to initialize parameters inside the focus

		return estimate(new Instance(model, model, focus));
	}

	/**
	 * TODO LP: Should try to make this function thread-safe. It may check the
	 * method that creates the estimation (which clones the model) and also the
	 * generate parameters method.
	 * 
	 * @param model
	 * @return
	 */
	Estimation estimate(Instance instance) {
		int retry = parameters.minimumRetryForNaN;

		// This loops to make sure that a non-NaN estimation is found unless it
		// has tried for the minimum number of times of retry
		do {
			chickeringHeckermanRestart(instance);
			secondStage(instance);

			try {

				// the following code may have become obsolete because some
				// measures have taken to prevent too small eigenvalues of
				// covariance matrices
				if (constrainer.detect(instance.best().model())) {
					String path =
							Log.writeTemporaryFile("spurious-model",
									instance.best());
					Log.errorWriter().format(
							"Spurious model detected: %s [%s]\n",
							instance.best().model().getName(), path);
					instance.best().setInvalidLoglikehood();
				} else if (Double.isNaN(instance.best().loglikelihood())) {
					String path =
							Log.writeTemporaryFile("NaN-model", instance.best());
					Log.errorWriter().format("NaN model detected: %s [%s]\n",
							instance.best().model().getName(), path);
				} else {
					// break if successful
					break;
				}
			} catch (CovarianceConstrainer.ImproperValueException e) {
				String path =
						Log.writeTemporaryFile(
								"ImproperEigenvalueDecomposition-model",
								instance.best());
				Log.errorWriter().format(
						"ImproperEigenvalueDecomposition model detected: %s [%s]\n",
						instance.best().model().getName(), path);
			}

			// retry if the estimation has a NaN loglikelihood
			instance.stepsRun = 0;
			retry--;
			instance.forceGenerateParameters = true;

		} while (retry > 0);

		if (Double.isNaN(instance.best().loglikelihood())) {
			String path =
					Log.writeTemporaryFile("retry-failed-model",
							instance.best());
			Log.errorWriter().format(
					"It failed to find a valid estimation without NaN: %s [%s]\n",
					instance.best().model().getName(), path);
		}

		return instance.best();
	}

	protected EstimationFactory estimationFactory() {
		return estimationFactory;
	}

	/**
	 * Sets the number of threads used in the estimation of EM.
	 * 
	 * In case more than one threads is specified, it uses a multithreading
	 * version of estimation. A fixed pool with the requested number of threads
	 * is used in this EM. However, note that different EM instances do not
	 * share these threads. Rather, only the same EM instance share these
	 * threads, so that there can be at most the specified number of threads in
	 * one estimation of EM.
	 * 
	 * @param threads
	 *            number of threads used in the estimation of EM.
	 * @param executor
	 *            executor holding the threads. It should be able to run at
	 *            least the specified number of threads. It is ignored if it has
	 *            only one thread.
	 */
	public void setMultithreading(int threads, ExecutorService executor) {
		if (threads == 1)
			estimationFactory = Estimation.FACTORY;
		else {
			estimationFactory =
					MultithreadingEstimation.createFactory(threads, executor);
		}
	}

	public void setSeed(long seed) {
	    this.seed = seed;
    }

	/**
	 * Initializes estimation instances for the EM algorithm. Each estimation
	 * may start from a different value.
	 * 
	 * @param size
	 *            number of estimations to create
	 * @param model
	 *            model to estimate
	 * @param focus
	 *            holds the parameters that need to be updated by EM
	 * @return array of initialized estimation
	 */
	protected abstract Estimation[] createEstimations(int size, Gltm model,
			Focus focus);

	private void step(Estimation estimation) {
		try {
			estimation.savePreviousLoglikelihood();
			VariableStatisticsMap map =
					estimation.collectSufficientStatistics();
			computeMlParameters(estimation.model(), map);
		} catch (ImpossibleEvidenceException e) {
			// discard this estimation when it has badly initialized parameters
			// that leads to some impossible data cases
			estimation.setInvalidLoglikehood();
		} catch (CovarianceConstrainer.ImproperValueException e) {
			// discard this estimation when it has badly initialized parameters
			// that leads to improper values for eigenvalue decomposition
			estimation.setInvalidLoglikehood();
		}
	}

	/**
	 * Compute the maximum likelihood parameters for the given model with the
	 * collected sufficient statistics.
	 */
	public void computeMlParameters(Gltm model, final VariableStatisticsMap map) {
		for (AbstractNode node : model.getNodes()) {
			BeliefNode beliefNode = (BeliefNode) node;

			// skip this node if its corresponding sufficient statistics is not
			// found

			// compute the potentials for each of the belief node
			beliefNode.accept(new BeliefNode.Visitor<Void>() {

				public Void visit(ContinuousBeliefNode node) {
					CGPotential potential = node.potential();
					MixedCliqueSufficientStatistics statistics =
							map.get(node.getVariable());

					if (statistics == null)
						return null;

					node.setPotential(statistics.computePotential(
							node.getVariable(), potential.discreteVariable()));
					constrainer.adjust(node.potential());
					return null;
				}

				public Void visit(DiscreteBeliefNode node) {
					// for discrete node, the statistics can be discrete or
					// mixed
					SufficientStatistics statistics =
							map.get(node.getVariable());

					if (statistics == null)
						return null;

					node.setCpt(statistics.computePotential(node.getVariable(),
							node.getDiscreteParentVariables()));
					return null;
				}
			});
		}

	}

	/**
	 * Initializes (randomizes) the parameters of the model. If the focus is
	 * specified, only the part of model under focus is initialized.
	 * 
	 * @param model
	 * @param focus
	 */
	private void initializeParameters(Gltm model, Focus focus) {
		for (AbstractNode node : model.getNodes()) {
			BeliefNode beliefNode = (BeliefNode) node;
			if (focus == null || focus.contains(beliefNode.getVariable()))
				generator.generate(beliefNode);
		}
	}

	/**
	 * The best candidate should be put at the first item of the candidates
	 * array after this method call.
	 * 
	 * @param instance
	 */
	private void chickeringHeckermanRestart(Instance instance) {

		instance.candidates =
				createEstimations(parameters.restarts, instance.model,
						instance.focus);
		instance.size = instance.candidates.length;

		for (int i = 0; i < instance.candidates.length; i++) {
			if (i != 0 || !parameters.reuseParameters
					|| instance.forceGenerateParameters)
				initializeParameters(instance.candidates[i].model(),
						instance.focus);
		}

		// We run several steps of emStep before killing starting points for two
		// reasons: 1. the loglikelihood computed is always that of previous
		// model. 2. When reuse, the reused model is kind of dominant because
		// maybe it has already EMed.
		repeatSteps(instance, parameters.initialIterations, false);

		// in each round, half of the candidates are eliminated
		int stepsPerRound = 1;

		while (instance.size > 1 && instance.stepsRun < parameters.maxSteps) {
			repeatSteps(instance, stepsPerRound, true);

			Arrays.sort(instance.candidates, 0, instance.size,
					Estimation.LOGLIKELIHOOD_COMPARATOR);

			// removeExtremeEstimations(instance);

			// remove half of the candidates, and release the reference to them
			// to free up memory
			int newSize = instance.size / 2;
			Arrays.fill(instance.candidates, newSize, instance.size, null);
			instance.size = newSize;

			// doubles EM steps subject to maximum step constraint
			stepsPerRound =
					Math.min(stepsPerRound * 2, parameters.maxSteps
							- instance.stepsRun);
		}

		// the estimation at 0 is the best one
	}

	/**
	 * Runs the second stage on single estimation of model. Updates the
	 * parameters until it has met the termination criteria.
	 * 
	 * @param estimation
	 */
	private void secondStage(Instance instance) {
		final Estimation current = instance.best();

		int lastStep =
				parameters.secondStageSteps > 0 ? parameters.secondStageSteps
						+ instance.stepsRun : parameters.maxSteps;

		do {
			if (current.improvement() < parameters.threshold)
				break;

			// TODO LP: should we check the number of steps before this is run?
			step(current);
			instance.stepsRun++;
		} while (instance.stepsRun < lastStep);
	}

	/**
	 * Runs the EM for a number of steps, on an array of propagation instances.
	 * 
	 * @param propagations
	 * @param length
	 *            length of the first propagations that are run, others are
	 *            ignored
	 * @param steps
	 *            number of steps to run
	 * @param considerThreshold
	 *            whether to consider the threshold to break the computation
	 *            earlier
	 */
	private void repeatSteps(Instance instance, int steps,
			boolean considerThreshold) {
		int maxSteps = 0;

		for (int i = 0; i < instance.size; i++) {
			Estimation candidate = instance.candidates[i];

			int step = 0;
			for (step = 0; step < steps; step++) {
				// break if the likelihood improvement falls under the
				// threshold, which empirically may shorten the running by 15%.
				if (candidate.improvement() > parameters.threshold)
					step(candidate);
				else
					break;
			}

			maxSteps = Math.max(maxSteps, step);
		}

		instance.stepsRun += maxSteps;
	}

	/**
	 * Removes the estimations that have extreme values of likelihood. Those
	 * estimations are likely to be spurious solutions.
	 * 
	 * @param estimations
	 *            should be sorted by likelihood
	 * @param size
	 *            number of estimations under consideration, starting from the
	 *            first item
	 */
	@SuppressWarnings("unused")
	private void removeExtremeEstimations(Instance instance) {
		// TODO LP: should improve this method

		NormalSufficientStatistics statistics =
				new NormalSufficientStatistics(1);

		DoubleMatrix1D likelihoodMatrix = new DenseDoubleMatrix1D(1);

		for (int i = 0; i < instance.size; i++) {
			double loglikelihood = instance.candidates[i].loglikelihood();
			if (Double.isNaN(loglikelihood))
				continue;

			likelihoodMatrix.set(0, loglikelihood);
			statistics.add(likelihoodMatrix, 1);
		}

		double mean = statistics.computeMean().get(0);
		double sd = Math.sqrt(statistics.computeCovariance().get(0, 0));
		double threshold = sd * 4;
		if (threshold == 0)
			return;

		for (int i = 0; i < instance.size; i++) {
			double loglikelihood = instance.candidates[i].loglikelihood();
			if (Double.isNaN(loglikelihood))
				continue;

			double diff = loglikelihood - mean;
			if (diff > threshold) {
				// if the likelihood is too far away from mean, it is likely to
				// be spurious value

				instance.extremeDetected = true;

				String path =
						Log.writeTemporaryFile("extreme-model",
								instance.candidates[i]);
				Log.errorWriter().format(
						"Model with extreme likelihood detected: %s [%s]\n",
						instance.candidates[i].model().getName(), path);

				instance.candidates[i].setInvalidLoglikehood();
			}
		}
	}

	/**
	 * Logs the description of this EM algorithm.
	 * 
	 * @param writer
	 * @param purpose
	 *            purpose of this EM
	 */
	public void writeXml(PrintWriter writer, String purpose) {
		writer.format("<em name='%s' purpose='%s' %s/>", name(), purpose,
				parameters.xmlAttributes());
		writer.println();
	}

	public String name() {
		return getClass().getSimpleName();
	}

	public CovarianceConstrainer constrainer() {
		return constrainer;
	}
}
