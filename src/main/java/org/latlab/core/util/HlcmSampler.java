package org.latlab.core.util;

import org.latlab.core.graph.DirectedNode;
import org.latlab.core.model.DiscreteBeliefNode;
import org.latlab.core.model.DiscreteVariableExtractor;
import org.latlab.core.model.HLCM;

import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * It generates samples from a HLCM. The samples consist of data cases for the
 * observed variables.
 * 
 * @author leonard
 * 
 */
public class HlcmSampler {
	public HlcmSampler(HLCM model) {
		this.model = model;
	}

	public DataSet sample(int numberOfCases) {
		DataSet data = createEmptyDataSet();
		DiscreteBeliefNode root = getRoot();
		Map<DiscreteVariable, Integer> map = data.createVariableToIndexMap();

		for (int i = 0; i < numberOfCases; i++) {
			data.addDataCase(sampleCase(root, map), 1);
		}

		return data;
	}

	void setSeed(long seed) {
		random.setSeed(seed);
	}

	private DiscreteBeliefNode getRoot() {
		List<DiscreteBeliefNode> roots =
			Algorithm.filter(
				model.getNodes(), new Caster<DiscreteBeliefNode>(),
				DirectedNode.ROOT_PREDICATE);

		if (roots.size() != 1)
			throw new IllegalArgumentException(
				"The model does not have one and only one root.");

		return roots.get(0);
	}

	private List<DiscreteVariable> getObservedVariables() {
		List<DiscreteBeliefNode> observedNodes =
			Algorithm.filter(
				model.getNodes(), new Caster<DiscreteBeliefNode>(),
				DirectedNode.LEAF_PREDICATE);
		return Algorithm.convert(observedNodes, new DiscreteVariableExtractor());
	}

	private DataSet createEmptyDataSet() {
		List<DiscreteVariable> observedVariables = getObservedVariables();
		return new DataSet(
			observedVariables.toArray(new DiscreteVariable[observedVariables.size()]));
	}

	private int[] sampleCase(
		DiscreteBeliefNode root, Map<DiscreteVariable, Integer> variableIndexMap) {
		int[] dataCase = new int[variableIndexMap.size()];
		Function prior = root.potential();
		sampleCaseRecursive(dataCase, root, prior, variableIndexMap);
		return dataCase;
	}

	private void sampleCaseRecursive(
		int[] dataCase, DiscreteBeliefNode node, Function prior,
		Map<DiscreteVariable, Integer> variableIndexMap) {
		int state = sample(prior);

		List<DiscreteBeliefNode> children =
			Algorithm.castTo(node.getChildren(), DiscreteBeliefNode.class);

		if (children.size() == 0) {
			int index = variableIndexMap.get(node.getVariable());
			dataCase[index] = state;
		} else {
			for (DiscreteBeliefNode child : children) {
				Function cpt = child.potential();
				Function childPrior = cpt.project(node.getVariable(), state);
				sampleCaseRecursive(
					dataCase, child, childPrior, variableIndexMap);
			}
		}
	}

	private int sample(Function prior) {
		assert prior.getDimension() == 1;
		double[] values = prior.getCells(prior.getVariables());
		double r = random.nextDouble();

		double cummulative = 0;
		for (int i = 0; i < prior.getDomainSize(); i++) {
			cummulative += values[i];
			if (r < cummulative)
				return i;
		}

		return prior.getDomainSize() - 1;
	}

	private final HLCM model;
	private Random random = new Random();
}
