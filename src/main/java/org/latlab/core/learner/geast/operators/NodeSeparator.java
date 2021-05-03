package org.latlab.core.learner.geast.operators;

import org.latlab.core.graph.AbstractNode;
import org.latlab.core.learner.geast.Estimation;
import org.latlab.core.learner.geast.IModelWithScore;
import org.latlab.core.learner.geast.context.ISearchOperatorContext;
import org.latlab.core.model.ContinuousBeliefNode;
import org.latlab.core.util.*;

import java.util.Collection;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

public class NodeSeparator extends SearchOperatorImpl {

	public static class Candidate extends
			org.latlab.core.learner.geast.operators.SearchCandidate {
		private final static String ELEMENT = "separateNode";
		private static final String OPERATOR_NAME = "NS";
		private final static String ATTRIBUTES_FORMAT =
				"source='%s' separated='%s'";

		private final JointContinuousVariable source;
		private final Collection<SingularContinuousVariable> separated;

		protected Candidate(IModelWithScore base,
				JointContinuousVariable source,
				SingularContinuousVariable separated) {
			this(base, source, Collections.singleton(separated));
		}

		protected Candidate(IModelWithScore base,
				JointContinuousVariable source,
				Collection<SingularContinuousVariable> separated) {
			super(base);
			this.source = source;
			this.separated = separated;

			Pair<ContinuousBeliefNode, ContinuousBeliefNode> newNodes =
					structure.separate(true, source, separated);

			modification.add(newNodes.first.getVariable());
			modification.add(newNodes.second.getVariable());
		}

		@Override
		public String attributes() {
			return String.format(ATTRIBUTES_FORMAT, source.getName(),
					Variable.getName(separated, ", "));
		}

		@Override
		public String element() {
			return ELEMENT;
		}

		@Override
		public String name() {
			return "NodeSeparationCandidate";
		}

		@Override
		public String operatorName() {
			return OPERATOR_NAME;
		}
	}

	public NodeSeparator(ISearchOperatorContext context) {
		super(context);
	}

	/**
	 * Generate candidates by separating one node from the each of the pouch.
	 */
	@Override
	protected LinkedList<SearchCandidate> generateCandidates(
			IModelWithScore base) {
		LinkedList<SearchCandidate> candidates =
				new LinkedList<SearchCandidate>();

		List<AbstractNode> continuousNodes =
				Algorithm.filter(base.model().getNodes(),
						new InstanceOfPredicate<AbstractNode>(
								ContinuousBeliefNode.class));

		for (AbstractNode n : continuousNodes) {
			ContinuousBeliefNode node = (ContinuousBeliefNode) n;
			int numberOfVariables = node.getVariable().variables().size();
			if (numberOfVariables <= 1)
				continue;
			else if (numberOfVariables == 2) {
				candidates.add(new Candidate(base, node.getVariable(),
						node.getVariable().variables().iterator().next()));
			} else {
				for (SingularContinuousVariable separated : node.getVariable().variables()) {
					candidates.add(new Candidate(base, node.getVariable(),
							separated));
				}
			}
		}

		return candidates;
	}

	@SuppressWarnings("unused")
	private LinkedList<SearchCandidate> generateCandidatesFromAllCombinations(
			Estimation base) {
		LinkedList<SearchCandidate> candidates =
				new LinkedList<SearchCandidate>();

		List<AbstractNode> continuousNodes =
				Algorithm.filter(base.model().getNodes(),
						new InstanceOfPredicate<AbstractNode>(
								ContinuousBeliefNode.class));

		// TODO LP - should consider not enumerating all the possible
		// combinations. Instead, may consider separating one variable from
		// a pouch node, and then consider moving any variable from the original
		// pouch node to the new node.

		for (AbstractNode n : continuousNodes) {
			ContinuousBeliefNode node = (ContinuousBeliefNode) n;
			int numberOfVariables = node.getVariable().variables().size();
			if (numberOfVariables <= 1)
				continue;

			CombinationGenerator<SingularContinuousVariable> generator =
					new CombinationGenerator<SingularContinuousVariable>(
							node.getVariable().variables());

			int max = numberOfVariables / 2;

			for (int size = 1; size <= max; size++) {
				for (List<SingularContinuousVariable> separated : generator.generate(size)) {
					SearchCandidate candidate =
							new Candidate(base, node.getVariable(), separated);
					candidates.add(candidate);
				}
			}

			// TODO LP: When size is half the number of variables, consider a
			// set {a,b}. It generates two candidates with separation sets {a}
			// and {b}, but they lead to the same candidate. Later can find a
			// way to eliminate the redundant sets.
		}

		return candidates;
	}

}
