package org.latlab.core.learner.geast;

import org.latlab.core.data.MixedDataSet;
import org.latlab.core.learner.geast.context.Context;
import org.latlab.core.learner.geast.procedures.Procedure;
import org.latlab.core.learner.geast.procedures.StateIntroductionProcedure;
import org.latlab.core.model.Builder;
import org.latlab.core.model.Gltm;

/**
 * Learns finite mixture model.
 * 
 * @author leonard
 * 
 */
public class FmmLearner {
	private final Geast geast;
	private final int initialNumberOfClusters;

	/**
	 * 
	 * @param data
	 * @param log
	 * @param initial
	 * @param increase
	 *            whether to increase the number of states
	 */
	public FmmLearner(MixedDataSet data, Log log, int initial, boolean increase) {
		this(Geast.DEFAULT_THREADS, Geast.DEFAULT_THRESHOLD, data, log,
				new FullEm(data, true, 64, 500, 0.01), initial, increase);
	}

	/**
	 * 
	 * @param threads
	 * @param screening
	 * @param threshold
	 * @param data
	 * @param log
	 * @param em
	 * @param initial
	 * @param increase
	 */
	public FmmLearner(int threads, double threshold, MixedDataSet data,
			Log log, EmFramework em, int initial, boolean increase) {
		Context context = new Context(threads, Geast.DEFAULT_SCREENING,
				threshold, data, log, em, em, em);
		Procedure[] procedures = increase ? new Procedure[] { new StateIntroductionProcedure(
				context) }
				: new Procedure[] {};
		geast = new Geast(context, procedures);

		this.initialNumberOfClusters = initial;
	}

	public IModelWithScore learn() {
		Gltm initial = Builder.buildMixedMixtureModel(new Gltm(),
				initialNumberOfClusters,
				geast.context().data().getNonClassVariables());
		return geast.learn(initial);
	}

	public void setCommandLine(String commandLine) {
		geast.commandLine = commandLine;
	}
}
