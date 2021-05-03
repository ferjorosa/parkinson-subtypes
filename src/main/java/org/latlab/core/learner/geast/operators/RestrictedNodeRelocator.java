package org.latlab.core.learner.geast.operators;

import org.latlab.core.graph.DirectedNode;
import org.latlab.core.learner.geast.IModelWithScore;
import org.latlab.core.learner.geast.context.ISearchOperatorContext;
import org.latlab.core.model.BeliefNode;
import org.latlab.core.util.DiscreteVariable;
import org.latlab.core.util.Variable;

import java.util.LinkedList;

/**
 * Generates candidates by consider moving any child from connecting to the
 * given original node to become a child of the given destination node.
 * 
 * @author leonard
 * 
 */
public class RestrictedNodeRelocator extends SearchOperatorImpl {

	public static class Candidate extends NodeRelocator.Candidate {
		private static final String ATTRIBUTES_FORMAT =
				"node='%s' origin='%s' destination='%s'";
		private static final String ELEMENT = "restrictedRelocateNode";

		protected Candidate(IModelWithScore base, Variable moving,
				DiscreteVariable destination) {
			super(base, base.model(), moving, destination);
		}

		@Override
		public String attributes() {
			return String.format(ATTRIBUTES_FORMAT, moving.getName(),
					origin.getName(), destination.getName());
		}

		@Override
		public String element() {
			return ELEMENT;
		}
	}

	private final DiscreteVariable origin;
	private final DiscreteVariable destination;

	public RestrictedNodeRelocator(ISearchOperatorContext context,
			DiscreteVariable origin, DiscreteVariable destination) {
		super(context);
		this.origin = origin;
		this.destination = destination;
	}

	@Override
	protected LinkedList<SearchCandidate> generateCandidates(
			IModelWithScore base) {
		LinkedList<SearchCandidate> candidates =
				new LinkedList<SearchCandidate>();

		if (base.model().getNode(origin).getDegree() < 4)
			return new LinkedList<SearchCandidate>();

		for (DirectedNode child : base.model().getNode(origin).getChildren()) {
			Variable childVariable = ((BeliefNode) child).getVariable();

			if (childVariable == destination)
				continue;

			Candidate candidate =
					new Candidate(base, childVariable, destination);

			if (!candidate.model().hasRegularCardinality(childVariable))
				continue;

			candidates.add(candidate);
		}

		return candidates;
	}

}
