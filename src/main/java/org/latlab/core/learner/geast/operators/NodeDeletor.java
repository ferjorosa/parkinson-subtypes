package org.latlab.core.learner.geast.operators;

import org.latlab.core.graph.AbstractNode;
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

public class NodeDeletor extends SearchOperatorImpl {

	/**
	 * Deletes the target variable and put all the neighbors of the target
	 * variable to the dock variable as children.
	 * 
	 * @author leonard
	 * 
	 */
	public static class Candidate extends
			org.latlab.core.learner.geast.operators.SearchCandidate {

		private static final String ELEMENT = "deleteNode";
		private static final String OPERATOR_NAME = "ND";
		private final static String ATTRIBUTES_FORMAT = "node='%s' dock='%s'";

		public final DiscreteVariable target;
		public final Variable dock;

		public String element() {
			return ELEMENT;
		}

		protected Candidate(IModelWithScore base, DiscreteVariable target,
				DiscreteVariable dock) {
			super(base);
			this.target = target;
			this.dock = dock;

			BeliefNode targetNode = structure.getNode(target);
			BeliefNode dockNode = structure.getNode(dock);

			List<DirectedNode> parents =
					new ArrayList<DirectedNode>(targetNode.getParents());
			List<DirectedNode> children =
					new ArrayList<DirectedNode>(targetNode.getChildren());

			structure.removeNode(targetNode);

			for (DirectedNode parentNode : parents) {
				if (parentNode != dockNode) {
					structure.addEdge(dockNode, parentNode);
					modification.add(dockNode.getVariable());
				}
			}
			for (DirectedNode childNode : children) {
				if (childNode != dockNode) {
					structure.addEdge(childNode, dockNode);
					modification.add(((BeliefNode) childNode).getVariable());
				}
			}
		}

		/**
		 * For testing.
		 */
		public static Candidate construct(IModelWithScore base,
				DiscreteVariable target, DiscreteVariable dock) {
			return new Candidate(base, target, dock);
		}

		@Override
		public String attributes() {
			return String.format(ATTRIBUTES_FORMAT, target.getName(),
					dock.getName());
		}

		@Override
		public String name() {
			return "NodeDeletionCandidate";
		}

		@Override
		public String operatorName() {
			return OPERATOR_NAME;
		}

	}

	public NodeDeletor(ISearchOperatorContext context) {
		super(context);
	}

	@Override
	protected LinkedList<SearchCandidate> generateCandidates(
			IModelWithScore base) {
		LinkedList<SearchCandidate> candidates =
				new LinkedList<SearchCandidate>();

		// List<DiscreteBeliefNode> internalNodes =
		// base.model().getInternalNodes();
		// if (internalNodes.size() <= 1)
		// return candidates;

		for (DiscreteBeliefNode target : base.model().getInternalNodes()) {
			for (AbstractNode n : target.getNeighbors()) {
				BeliefNode neighbor = (BeliefNode) n;

				// skip it if it is a observed node, since it can't hold the
				// other children
				if (neighbor.isLeaf())
					continue;

				Candidate candidate =
						new Candidate(base, target.getVariable(),
								(DiscreteVariable) neighbor.getVariable());
				candidates.add(candidate);
			}
		}

		return candidates;
	}

}
