package org.latlab.core.learner.geast.procedures;

import org.latlab.core.learner.geast.IModelWithScore;
import org.latlab.core.learner.geast.context.IProcedureContext;
import org.latlab.core.learner.geast.operators.GivenCandidate;
import org.latlab.core.learner.geast.operators.SearchCandidate;
import org.latlab.core.learner.geast.operators.SearchOperator;
import org.latlab.core.util.Evaluator;

/**
 * Refinement procedure, used in an expand procedure. It uses the specified
 * operator to iteratively search for a candidate model until the unit
 * improvement is smaller than the specified base score.
 * 
 * @author leonard
 * 
 */
public class RefinementProcedure implements Procedure {

	private final String name;
	private final IProcedureContext context;
	private final SearchOperator operator;
	private final Evaluator<SearchCandidate> evaluator;
	private final double baseScore;

	private boolean succeeded = false;

	public RefinementProcedure(String name, IProcedureContext context,
			SearchOperator operator, Evaluator<SearchCandidate> evaluator,
			double baseScore) {
		this.context = context;
		this.operator = operator;

		this.name = name;
		this.evaluator = evaluator;
		this.baseScore = baseScore;
	}

	public String name() {
		return name;
	}

	public SearchCandidate run(IModelWithScore base) {
		context.log().writeStartElementWithTime(name(), null);

		SearchCandidate current = new GivenCandidate(base);
		boolean stop = false;
		succeeded = false;
		double lastScore = baseScore;

		do {
			SearchCandidate candidate =
					operator.search(current.estimation(), evaluator);
			operator.update(candidate);

			context.log().writeElement("refining", candidate, true);

			if (candidate.isNew() && candidate.score() > lastScore) {
				current = candidate;
				lastScore = candidate.score();
				succeeded = true;

				context.log().writeElementWithCandidateToFile("step",
						candidate, true);
			} else {
				stop = true;
			}
		} while (!stop);

		context.log().writeElement("completed", current, false);
		context.log().writeEndElement(name());

		return current;

	}

	public boolean succeeded() {
		return succeeded;
	}
}
