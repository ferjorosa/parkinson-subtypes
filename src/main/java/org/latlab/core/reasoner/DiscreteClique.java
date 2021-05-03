package org.latlab.core.reasoner;

import org.latlab.core.util.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * Note: The potential has to be assigned before it can be used.
 * 
 * @author leonard
 * 
 */
public class DiscreteClique extends Clique {
	private CliquePotential<Function> potential;

	/**
	 * Holds the list of variables for this clique. This will be different from
	 * those variables in the potential when some of them have become evidence
	 * variables.
	 */
	private List<DiscreteVariable> variables;

	public DiscreteClique(NaturalCliqueTree tree, String name,
			List<DiscreteVariable> variables) {
		super(tree, name);

		this.variables = new ArrayList<DiscreteVariable>(variables);
	}

	/**
	 * Returns an unmodified list of the variables contained in this clique.
	 * 
	 * @return an unmodified list of the variables contained in this clique
	 */
	@Override
	public List<Variable> variables() {
		return new BaseTypeListView<Variable>(variables);
	}

	@Override
	public List<DiscreteVariable> discreteVariables() {
		return Collections.unmodifiableList(variables);
	}

	@Override
	public Function potential() {
		return potential == null? null : potential.content;
	}

	@Override
	public double logNormalization() {
		return potential.logNormalization;
	}

	@Override
	protected void addLogNormalization(double logNormalization) {
		potential.logNormalization += logNormalization;
	}

	@Override
	public boolean contains(Variable variable) {
		return variables.contains(variable);
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();

		builder.append(String.format("Variables: %s\n", Variable.getName(
				variables, ",")));

		builder.append(super.toString());
		return builder.toString();
	}

	@Override
	public Message computeMessage(Message multiplier, Separator separator,
			Set<DiscreteVariable> retainingVariables) {
		Function function =
				multiplier == null ? potential.content
						: potential.content.times(multiplier.function);

		for (DiscreteVariable variable : potential.content.getVariables()) {
			if (variable == separator.variable())
				continue;
			if (retainingVariables != null
					&& retainingVariables.contains(variable))
				continue;

			function = function.sumOut(variable);
		}

		double v =
				multiplier == null ? logNormalization() : logNormalization()
						+ multiplier.logNormalization();

		return new Message(function, v);
	}

	@Override
	public void reset() {
		potential = null;
	}

	@Override
	public void combine(Potential other, double logNormalization) {
		if (potential == null) {
			Function f = other.function().clone();
			potential = new CliquePotential<Function>(f, logNormalization);
		} else {
			Function f = potential.content.times(other);
			potential =
					new CliquePotential<Function>(f, logNormalization()
							+ logNormalization);
		}

		if (pivot) {
			normalize(Double.NaN);
		}
	}

	@Override
	public <T> T accept(Visitor<T> visitor) {
		return visitor.visit(this);
	}

}
