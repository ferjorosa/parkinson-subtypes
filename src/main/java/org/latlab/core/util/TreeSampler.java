package org.latlab.core.util;

import cern.colt.matrix.DoubleMatrix1D;
import org.latlab.core.data.MixedDataSet;
import org.latlab.core.data.io.ArffWriter;
import org.latlab.core.graph.DirectedNode;
import org.latlab.core.io.ParseException;
import org.latlab.core.io.bif.BifParser;
import org.latlab.core.model.*;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * It generates samples from a tree model. The samples consist of data cases for
 * the observed variables.
 * 
 * <p>
 * The tree model should have a discrete belief node as a root.
 * 
 * @author leonard
 * 
 */
public class TreeSampler {

	private Random random = new Random();
	private final Normal normal = new Normal();

	private final Gltm model;
	private final BeliefNode root;
	private final List<Variable> observedVariables;

	private MixedDataSet data;

	/**
	 * Option for included variables in the samples
	 * 
	 * @author leonard
	 * 
	 */
	public static enum VariableOption {
		ONLY_OBSERVED_VARIABLES, OBSERVED_VARIABLES_WITH_CLASS, ALL_VARIABLES
	};

	public static void main(String[] args) throws ParseException, IOException {
		// args[0] model file
		// args[1] data file
		// args[2] sample size
		// args[3] with class?

		if (args.length < 4) {
			System.out.println("java TreeSampler model_file data_file "
					+ "sample_size latent_variable_option {all, class, none}");
			return;
		}

		BifParser parser;
		parser = new BifParser(new FileInputStream(args[0]));
		Gltm model = parser.parse(new Gltm());

		TreeSampler sampler = new TreeSampler(model);

		VariableOption option =
				args[3].equals("all") ? VariableOption.ALL_VARIABLES
						: args[3].equals("class") ? VariableOption.OBSERVED_VARIABLES_WITH_CLASS
								: VariableOption.OBSERVED_VARIABLES_WITH_CLASS;
		
		MixedDataSet data = sampler.sample(Integer.parseInt(args[2]), option);
		ArffWriter.write(args[1], data);
		ArffWriter.write(args[1], data);
	}

	public TreeSampler(Gltm model) {
		this.model = model;
		root = model.getRoot();
		observedVariables = model.getLeafSingularVariables();
	}

	public MixedDataSet sample(int numberOfCases, boolean withClass) {
		VariableOption option =
				withClass ? VariableOption.OBSERVED_VARIABLES_WITH_CLASS
						: VariableOption.ONLY_OBSERVED_VARIABLES;

		return sample(numberOfCases, option);
	}

	public MixedDataSet sample(int numberOfCases, VariableOption option) {
		List<Variable> variables;

		switch (option) {
		case ALL_VARIABLES:
			variables = new ArrayList<Variable>(model.getVariables());
			break;
		case OBSERVED_VARIABLES_WITH_CLASS:
			variables = new ArrayList<Variable>(observedVariables.size() + 1);
			variables.addAll(observedVariables);
			variables.add(root.getVariable());
			break;
		case ONLY_OBSERVED_VARIABLES:
			variables = observedVariables;
			break;
		default:
			variables = Collections.emptyList();
		}

		data = MixedDataSet.createEmpty(variables, numberOfCases);

		for (int i = 0; i < numberOfCases; i++) {
			data.add(1, sampleCase((DiscreteBeliefNode) root));
		}

		return data;
	}

	void setSeed(long seed) {
		random.setSeed(seed);
	}

	private double[] sampleCase(DiscreteBeliefNode root) {
		double[] values = new double[data.variables().size()];

		Function prior = root.potential();
		int state = sample(prior);

		Integer index = data.indexOf(root.getVariable());
		if (index != null)
			values[index] = state;

		for (DirectedNode child : root.getChildren()) {
			sampleCaseRecursive((BeliefNode) child, root.getVariable(), state,
					values);
		}

		return values;
	}

	private void sampleCaseRecursive(BeliefNode node,
			final DiscreteVariable parent, final int parentState,
			final double[] values) {

		node.accept(new BeliefNode.Visitor<Void>() {

			public Void visit(DiscreteBeliefNode node) {
				Function prior = node.potential().project(parent, parentState);
				int state = sample(prior);
				Integer index = data.indexOf(node.getVariable());
				if (index != null)
					values[index] = state;

				for (DirectedNode child : node.getChildren()) {
					sampleCaseRecursive((BeliefNode) child, node.getVariable(),
							state, values);
				}

				return null;
			}

			public Void visit(ContinuousBeliefNode node) {
				CGPotential potential = node.potential();
				CGParameter parameter = potential.get(parentState);
				DoubleMatrix1D sample = sample(parameter);
				for (int i = 0; i < potential.continuousVariables().size(); i++) {
					SingularContinuousVariable variable =
							potential.continuousVariables().get(i);
					Integer index = data.indexOf(variable);
					if (index != null)
						values[index] = sample.getQuick(i);
				}

				return null;
			}

		});

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

	private DoubleMatrix1D sample(CGParameter parameter) {
		return normal.generateWith(parameter.A, parameter.C);
	}
}
