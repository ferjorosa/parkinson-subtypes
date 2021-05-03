package org.latlab.core.learner.geast.procedures;

import org.latlab.core.learner.geast.Geast;
import org.latlab.core.learner.geast.IModelWithScore;
import org.latlab.core.learner.geast.operators.SearchCandidate;

/**
 * Interface of a procedure for using in the {@link Geast} algorithm. It is used
 * to improve a base model and its estimation.
 * 
 * <p>
 * A procedure is assumed to be run once only.
 * 
 * @author leonard
 * 
 */
public interface Procedure {
	
	/**
	 * It runs the procedure and tries to find a model better than the current
	 * {@code base}. It should return an estimation with score at least as good
	 * as the given model.
	 * 
	 * @param base
	 * @return
	 */
	public abstract SearchCandidate run(final IModelWithScore base);

	/**
	 * Returns whether this procedure has succeeded to find a better model. It
	 * is initially set to {@code true}.
	 * 
	 * @return whether a better model has been found
	 */
	public abstract boolean succeeded();

	/**
	 * Name of this procedure.
	 * 
	 * @return name of this procedure
	 */
	public String name();
}