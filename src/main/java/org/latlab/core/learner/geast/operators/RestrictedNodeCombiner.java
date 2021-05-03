package org.latlab.core.learner.geast.operators;

import org.latlab.core.graph.DirectedNode;
import org.latlab.core.learner.geast.IModelWithScore;
import org.latlab.core.learner.geast.context.ISearchOperatorContext;
import org.latlab.core.model.ContinuousBeliefNode;
import org.latlab.core.model.DiscreteBeliefNode;
import org.latlab.core.util.Algorithm;
import org.latlab.core.util.DiscreteVariable;
import org.latlab.core.util.InstanceOfPredicate;
import org.latlab.core.util.JointContinuousVariable;

import java.util.LinkedList;
import java.util.List;

/**
 * Generates candidates by combining two continuous nodes having the same parent
 * node. This operator is restricted to a given parent node.
 * 
 * @author leonard
 * 
 */
public class RestrictedNodeCombiner extends NodeCombiner {

	public class Candidate extends NodeCombiner.Candidate {
		private final static String ELEMENT = "restrictedCombineNodes";

		protected Candidate(IModelWithScore base,
				DiscreteVariable parentVariable,
				JointContinuousVariable variable1,
				JointContinuousVariable variable2) {
			super(base, parentVariable, variable1, variable2);
		}

		@Override
		public String element() {
			return ELEMENT;
		}

		@Override
		public String name() {
			return "RestrictedNodeCombinationCandidate";
		}
	}

	private final DiscreteVariable parent;
	private JointContinuousVariable combined;

	public RestrictedNodeCombiner(ISearchOperatorContext context,
			DiscreteVariable parent, JointContinuousVariable combined) {
		super(context);
		this.parent = parent;
		this.combined = combined;
	}

	@Override
	protected LinkedList<SearchCandidate> generateCandidates(
			IModelWithScore base) {
		LinkedList<SearchCandidate> candidates =
				new LinkedList<SearchCandidate>();
		DiscreteBeliefNode parentNode = base.model().getNode(parent);
		List<DirectedNode> continuousChildren =
				Algorithm.filter(parentNode.getChildren(),
						new InstanceOfPredicate<DirectedNode>(
								ContinuousBeliefNode.class));

		// combine every child with the pivot variable
		for (int i = 0; i < continuousChildren.size(); i++) {
			ContinuousBeliefNode child =
					(ContinuousBeliefNode) continuousChildren.get(i);

			if (child.getVariable() == combined) {
				continue;
			}

			candidates.add(new Candidate(base, parent, combined,
					child.getVariable()));
		}
		return candidates;
	}

	public void update(SearchCandidate latest) {
		if (!latest.isNew())
			return;

		// since a new variable is created after combination, it should update
		// the current combined variable.
		Candidate candidate = (Candidate) latest;
		combined = candidate.newVariable;
	}
}
