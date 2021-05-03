package org.latlab.core.learner.geast;

import org.latlab.core.data.MixedDataSet;
import org.latlab.core.learner.geast.SeparateTreePropagation.SharedData;
import org.latlab.core.model.Gltm;

/**
 * Implements local EM algorithm.
 * 
 * It uses one clique tree for each data. However, this set of clique trees are
 * shared by different estimations.
 * 
 * @author leonard
 * 
 */
public class LocalEm extends EmFramework {

	public LocalEm(MixedDataSet data, boolean reuseParameters, int restarts,
			int secondStageSteps, double threshold) {
		super(data);
		use(new EmParameters(reuseParameters, restarts, secondStageSteps,
				Integer.MAX_VALUE, threshold));
	}

	public LocalEm(MixedDataSet data) {
		this(data, true, 64, 50, 1e-2);
	}

	@Override
	protected Estimation[] createEstimations(int size, Gltm model, Focus focus) {
		// different estimations share the same data propagation object

		SharedData sharedData = estimationFactory().createSharedData(model,
				data, focus);

		Estimation[] estimations = new Estimation[size];
		for (int i = 0; i < estimations.length; i++) {
			estimations[i] = estimationFactory().createRestricted(model, data,
					sharedData, parameters.smoothing);
		}

		return estimations;
	}

}
