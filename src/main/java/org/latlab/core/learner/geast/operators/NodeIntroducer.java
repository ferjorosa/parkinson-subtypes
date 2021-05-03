/**
 * 
 */
package org.latlab.core.learner.geast.operators;

import org.latlab.core.graph.DirectedNode;
import org.latlab.core.learner.geast.IModelWithScore;
import org.latlab.core.learner.geast.context.ISearchOperatorContext;
import org.latlab.core.model.BeliefNode;
import org.latlab.core.model.DiscreteBeliefNode;
import org.latlab.core.util.DiscreteVariable;
import org.latlab.core.util.Variable;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * @author leonard
 * 
 */
public class NodeIntroducer extends SearchOperatorImpl {

	public abstract static class Candidate extends
			org.latlab.core.learner.geast.operators.SearchCandidate {

		private static final String ELEMENT = "introduceNode";
		private static final String OPERATOR_NAME = "NI";

		public final DiscreteVariable origin;
		public final DiscreteVariable introduced;

		public Candidate(IModelWithScore base, DiscreteVariable origin) {
			super(base);
			this.origin = origin;
			this.introduced = new DiscreteVariable(origin.getCardinality());

			assert !origin.getName().equals(introduced.getName());
		}

		public String element() {
			return ELEMENT;
		}

		@Override
		public String operatorName() {
			return OPERATOR_NAME;
		}
	}

	/**
	 * Introduces a new node N as mediator between a origin variable V, and its
	 * two children C1 and C2.
	 * 
	 * <p>
	 * The original model is like C1<-V->C2, the returned model is like
	 * C1<-N->C2, and V->N.
	 * 
	 */
	public static class ChildChildCandidate extends Candidate {

		public final Variable child1;
		public final Variable child2;

		private static final String ATTRIBUTES_FORMAT =
				"node='%s' child1='%s' child2='%s'";

		public ChildChildCandidate(IModelWithScore base,
				DiscreteVariable origin, Variable child1, Variable child2) {
			super(base, origin);

			this.child1 = child1;
			this.child2 = child2;

			BeliefNode childNode1 = structure.getNode(child1);
			BeliefNode childNode2 = structure.getNode(child2);
			DiscreteBeliefNode originNode = structure.getNode(origin);
			DiscreteBeliefNode newNode = structure.addNode(introduced);

			// disconnect the children from the node
			structure.removeEdge(originNode, childNode1);
			structure.removeEdge(originNode, childNode2);

			// connect the new node to the two disconnected children
			structure.addEdge(childNode1, newNode);
			structure.addEdge(childNode2, newNode);

			// connect the new node with the node
			structure.addEdge(newNode, originNode);

			modification.add(newNode.getVariable());
			modification.add(childNode1.getVariable());
			modification.add(childNode2.getVariable());
		}

		@Override
		public String attributes() {
			return String.format(ATTRIBUTES_FORMAT, origin.getName(),
					child1.getName(), child2.getName());
		}

		@Override
		public String name() {
			return "NodeIntroductionChildChildCandidate";
		}
	}

	/**
	 * Introduces a new node N as mediator between a variable V, and its parent
	 * P and its child C.
	 * 
	 * <p>
	 * The original model is like P->V->C, the returned model is like P->N->C,
	 * and N->V.
	 * 
	 */
	public static class ParentChildCandidate extends Candidate {

		public final Variable parent;
		public final Variable child;

		private static final String ATTRIBUTES_FORMAT =
				"node='%s' parent='%s' child='%s'";

		public ParentChildCandidate(IModelWithScore base,
				DiscreteVariable origin, Variable parent, Variable child) {
			super(base, origin);
			this.parent = parent;
			this.child = child;

			BeliefNode parentNode = structure.getNode(parent);
			BeliefNode childNode = structure.getNode(child);
			DiscreteBeliefNode originNode = structure.getNode(origin);
			DiscreteBeliefNode newNode = structure.addNode(introduced);

			// disconnect the parent and child from the origin node
			structure.removeEdge(originNode, parentNode);
			structure.removeEdge(originNode, childNode);

			// connect the new node to the two disconnected children
			structure.addEdge(newNode, parentNode);
			structure.addEdge(childNode, newNode);

			// connect the new node with the origin node
			structure.addEdge(originNode, newNode);

			modification.add(newNode.getVariable());
			modification.add(childNode.getVariable());
			modification.add(originNode.getVariable());
		}

		@Override
		public String attributes() {
			return String.format(ATTRIBUTES_FORMAT, origin.getName(),
					parent.getName(), child.getName());
		}

		@Override
		public String name() {
			return "NodeIntroductionParentChildCandidate";
		}
	}

	public NodeIntroducer(ISearchOperatorContext context) {
		super(context);
	}

	@Override
	protected LinkedList<SearchCandidate> generateCandidates(
			IModelWithScore base) {
		LinkedList<SearchCandidate> candidates =
				new LinkedList<SearchCandidate>();

		for (DiscreteBeliefNode node : base.model().getInternalNodes()) {
			List<DirectedNode> childNodes =
					new ArrayList<DirectedNode>(node.getChildren());

			// if there are only 3 neighbors, introduction of a node with
			// same cardinality does not improve the model
			if (node.getNeighbors().size() <= 3)
				continue;

			for (int i = 0; i < childNodes.size(); i++) {
				BeliefNode child1 = (BeliefNode) childNodes.get(i);
				// for every pair of different children
				for (int j = i + 1; j < childNodes.size(); j++) {
					BeliefNode child2 = (BeliefNode) childNodes.get(j);

					Candidate candidate =
							new ChildChildCandidate(base, node.getVariable(),
									child1.getVariable(), child2.getVariable());
					candidates.add(candidate);
				}

				// for every pair of child and parent
				BeliefNode parentNode = (BeliefNode) node.getParent();
				if (parentNode != null) {
					Candidate candidate =
							new ParentChildCandidate(base, node.getVariable(),
									parentNode.getVariable(),
									child1.getVariable());
					candidates.add(candidate);
				}
			}
		}

		return candidates;
	}
}
