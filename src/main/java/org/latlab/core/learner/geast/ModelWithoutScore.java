package org.latlab.core.learner.geast;

import org.latlab.core.model.Gltm;

/**
 * Holds a model which does not have any scores.
 * 
 * @author leonard
 * 
 */
public class ModelWithoutScore implements IModelWithScore {

	private final Gltm model;

	public ModelWithoutScore(Gltm model) {
		this.model = model;
	}

	/**
	 * Returns {@code Double.NaN}.
	 */
	public double BicScore() {
		return Double.NaN;
	}

	/**
	 * Returns {@code Double.NaN}.
	 */
	public double loglikelihood() {
		return Double.NaN;
	}

	public Gltm model() {
		return model;
	}

	public int messageCount() {
		return 0;
	}

	public Gltm origin() {
		return model;
	}

}
