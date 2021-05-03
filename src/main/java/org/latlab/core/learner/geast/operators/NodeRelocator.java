package org.latlab.core.learner.geast.operators;

import org.latlab.core.graph.AbstractNode;
import org.latlab.core.learner.geast.IModelWithScore;
import org.latlab.core.learner.geast.context.ISearchOperatorContext;
import org.latlab.core.model.BeliefNode;
import org.latlab.core.model.DiscreteBeliefNode;
import org.latlab.core.model.Gltm;
import org.latlab.core.model.ModelManipulator;
import org.latlab.core.util.Variable;

import java.util.LinkedList;

public class NodeRelocator extends SearchOperatorImpl {

	public static class Candidate extends
			org.latlab.core.learner.geast.operators.SearchCandidate {

		private static final String ATTRIBUTES_FORMAT =
				"node='%s' origin='%s' destination='%s'";
		private static final String ELEMENT = "relocateNode";
		private static final String OPERATOR_NAME = "NR";

		public final Variable moving;
		public final Variable origin;
		public final Variable destination;

		public final Variable variableForRegularityChecking;

		/**
		 * The {@code adjustedModel} is adjusted from the {@code base} model so
		 * that the moving node can be disconnected from its parent and then
		 * connect to the destination variable. A model can fulfill this
		 * requirement by changing the root to the destination node.
		 * 
		 * @param base
		 * @param adjustedModel
		 * @param moving
		 * @param destination
		 */
		protected Candidate(IModelWithScore base, Gltm adjustedModel,
				Variable moving, Variable destination) {
			super(base, adjustedModel.clone(), null);

			this.moving = moving;
			this.destination = destination;

			BeliefNode movingNode = structure.getNode(moving);
			BeliefNode rootNode = structure.getNode(destination);
			BeliefNode originNode = (BeliefNode) movingNode.getParent();
			this.origin = originNode.getVariable();

			structure.removeEdge(movingNode, originNode);
			structure.addEdge(movingNode, rootNode);

			// check if the original node still has any children, and purge
			// it if not
			BeliefNode current = originNode;
			while (current != null) {
				if (current.getChildren().size() <= 0) {
					BeliefNode parent = (BeliefNode) current.getParent();
					structure.removeNode(current);
					current = parent;
				} else {
					break;
				}
			}

			assert current != null;

			// TODO LP: more consideration and checking has to be taken here,
			// for example whether the root node becomes a dangling leaf node
			if (current != null)
				variableForRegularityChecking = current.getVariable();
			else
				variableForRegularityChecking = null;

			modification.add(moving);
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

		@Override
		public String name() {
			return "NodeRelocationCandidate";
		}

		@Override
		public String operatorName() {
			return OPERATOR_NAME;
		}
	}

	public NodeRelocator(ISearchOperatorContext context) {
		super(context);
	}

	@Override
	protected LinkedList<SearchCandidate> generateCandidates(
			IModelWithScore base) {
		LinkedList<SearchCandidate> candidates =
				new LinkedList<SearchCandidate>();

		Gltm baseModel = base.model();

		for (DiscreteBeliefNode destination : baseModel.getInternalNodes()) {
			// change the root to the destination for convenience, so that a
			// moving node must detach from its parent, and only the parameter
			// for the edge between the moving node and the destination node
			// has to be computed
			Gltm modelWithNewRoot = baseModel.clone();
			ModelManipulator.changeRoot(modelWithNewRoot,
					destination.getVariable());

			for (AbstractNode abstractNode : modelWithNewRoot.getNodes()) {
				BeliefNode node = (BeliefNode) abstractNode;

				// if the node is the destination node or the child of the
				// destination node, don't have to move
				if (node.isRoot() || node.getParent().isRoot())
					continue;

				Candidate candidate =
						new Candidate(base, modelWithNewRoot,
								node.getVariable(), destination.getVariable());

				if (candidate.variableForRegularityChecking == null
						|| !candidate.model().hasRegularCardinality(
								candidate.variableForRegularityChecking))
					continue;

				candidates.add(candidate);
			}
		}

		return candidates;
	}

}
