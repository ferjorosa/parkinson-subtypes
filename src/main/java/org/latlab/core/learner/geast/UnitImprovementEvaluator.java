package org.latlab.core.learner.geast;

import org.latlab.core.learner.geast.operators.SearchCandidate;
import org.latlab.core.model.Gltm;
import org.latlab.core.util.Evaluator;

/**
 * Evaluates a candidate based on its unit improvement over an estimation of a
 * base model. The unit improvement is the increase in BIC score divided by the
 * increase in dimensions over the base model.
 * 
 * @author leonard
 * 
 */
public class UnitImprovementEvaluator implements Evaluator<SearchCandidate> {

	private final int baseDimension;
	private final double baseBic;
	private final Gltm baseOrigin;

	public UnitImprovementEvaluator(IModelWithScore base) {
		baseDimension = base.model().computeDimension();
		baseBic = base.BicScore();
		this.baseOrigin = base.origin();
	}

	/**
	 * For testing purpose.
	 */
	UnitImprovementEvaluator(int dimension, double bic, Gltm origin) {
		baseDimension = dimension;
		baseBic = bic;
		baseOrigin = origin;
	}

	public double evaluate(SearchCandidate candidate) {
		IModelWithScore estimation = candidate.estimation();

		// if they share the same origin, it means it has zero
		// improvement
		if (estimation.origin() == baseOrigin)
			return 0;

		// the dimension shouldn't be negative, since it is in an
		// expansion procedure.
		int dimensionDifference = estimation.model().computeDimension()
				- baseDimension;
		assert dimensionDifference > 0;

		// the score should not be negative, since even if it can't find
		// any other model, the base model is given as argument to this
		// function.
		double scoreDifference = estimation.BicScore() - baseBic;
		assert scoreDifference >= 0;

		assert dimensionDifference != 0;
		return scoreDifference / dimensionDifference;
	}

}
