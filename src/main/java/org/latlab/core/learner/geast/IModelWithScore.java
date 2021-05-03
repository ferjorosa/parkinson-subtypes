package org.latlab.core.learner.geast;

import org.latlab.core.model.Gltm;

/**
 * Represents a model with its scores.
 * 
 * @author leonard
 * 
 */
public interface IModelWithScore {

	/**
	 * Returns the model.
	 * 
	 * @return model
	 */
	public Gltm model();

	/**
	 * Returns the log likelihood.
	 * 
	 * @return loglikelihood
	 */
	public double loglikelihood();

	/**
	 * Returns the BIC score.
	 * 
	 * @return BIC score
	 */
	public double BicScore();

	/**
	 * The model from which this model is estimated from. This model shares the
	 * same structure of the origin model but the parameters are updated after
	 * estimation.
	 * 
	 * @return origin model
	 */
	public Gltm origin();

	/**
	 * Returns the number of messages passed in the estimation of this model.
	 * For testing purpose.
	 * 
	 * @return number of messages passed in estimation
	 */
	public int messageCount();
}
