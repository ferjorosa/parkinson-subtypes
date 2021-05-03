package org.latlab.core.learner.geast.operators;

import org.latlab.core.graph.DirectedNode;
import org.latlab.core.learner.geast.IModelWithScore;
import org.latlab.core.learner.geast.context.ISearchOperatorContext;
import org.latlab.core.model.BeliefNode;
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
 * node.
 * 
 * @author leonard
 * 
 */
public class NodeCombiner extends SearchOperatorImpl {

	public class Candidate extends SearchCandidate {
		private final static String ELEMENT = "combineNodes";
		private final static String OPERATOR_NAME = "NC";
		private final static String ATTRIBUTES_FORMAT = "node1='%s' node2='%s'";

		public final JointContinuousVariable variable1;
		public final JointContinuousVariable variable2;
		public final JointContinuousVariable newVariable;
		public final DiscreteVariable parentVariable;

		protected Candidate(IModelWithScore base,
				DiscreteVariable parentVariable,
				JointContinuousVariable variable1,
				JointContinuousVariable variable2) {
			super(base);

			this.variable1 = variable1;
			this.variable2 = variable2;
			newVariable = new JointContinuousVariable(variable1, variable2);
			this.parentVariable = parentVariable;

			BeliefNode node1 = structure.getNode(variable1);
			BeliefNode node2 = structure.getNode(variable2);

			assert node1 != node2;

			DirectedNode parent = node1.getParent();
			assert parent == node2.getParent();

			structure.removeNode(node1);
			structure.removeNode(node2);

			// this has to be added after the remove node method, otherwise the
			// keys in the variable map would be removed by the removeNode
			// methods.
			BeliefNode newNode = structure.addNode(newVariable);
			structure.addEdge(newNode, parent);

			modification.add(newVariable);
		}

		@Override
		public String attributes() {
			return String.format(ATTRIBUTES_FORMAT, variable1.getName(),
					variable2.getName());
		}

		@Override
		public String element() {
			return ELEMENT;
		}

		@Override
		public String name() {
			return "NodeCombinationCandidate";
		}

		@Override
		public String operatorName() {
			return OPERATOR_NAME;
		}

	}

	public NodeCombiner(ISearchOperatorContext context) {
		super(context);
	}

	@Override
	protected LinkedList<SearchCandidate> generateCandidates(
			IModelWithScore base) {
		LinkedList<SearchCandidate> candidates =
				new LinkedList<SearchCandidate>();

		for (DiscreteBeliefNode node : base.model().getInternalNodes()) {
			List<DirectedNode> continuousChildren =
					Algorithm.filter(node.getChildren(),
							new InstanceOfPredicate<DirectedNode>(
									ContinuousBeliefNode.class));

			// combine every pair of continuous nodes belonging to the same
			// parent and use it as a candidate for improvement
			for (int i = 0; i < continuousChildren.size(); i++) {
				for (int j = i + 1; j < continuousChildren.size(); j++) {
					ContinuousBeliefNode child1 =
							(ContinuousBeliefNode) continuousChildren.get(i);
					ContinuousBeliefNode child2 =
							(ContinuousBeliefNode) continuousChildren.get(j);

					candidates.add(new Candidate(base, node.getVariable(),
							child1.getVariable(), child2.getVariable()));
				}
			}
		}

		return candidates;
	}
}
