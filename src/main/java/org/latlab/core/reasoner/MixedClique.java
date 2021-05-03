package org.latlab.core.reasoner;

import org.latlab.core.model.CGPotential;
import org.latlab.core.util.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * A clique for both continuous variables and discrete variables. Currently it
 * supports only one discrete variable.
 * 
 * <p>
 * Note: The potential has to be assigned before it can be used.
 * 
 * @author leonard
 * 
 */
public class MixedClique extends Clique {
	private CliquePotential<CGPotential> potential;
	private DiscreteVariable discreteVariable;
	private JointContinuousVariable jointVariable;

	public MixedClique(NaturalCliqueTree tree, String name,
			JointContinuousVariable joint, DiscreteVariable discrete) {
		super(tree, name);

		jointVariable = joint;
		discreteVariable = discrete;
	}

	public DiscreteVariable discreteVariable() {
		return discreteVariable;
	}

	public JointContinuousVariable jointVariable() {
		return jointVariable;
	}

	@Override
	public CGPotential potential() {
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
		return discreteVariable() == variable || jointVariable() == variable
				|| jointVariable().variables().contains(variable);
	}

	public void assign(CGPotential potential) {
		this.potential = new CliquePotential<CGPotential>(potential);
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();

		builder.append(String.format("Variables: %s, %s\n",
				jointVariable.getName(), discreteVariable.getName()));

		builder.append(super.toString());
		return builder.toString();
	}

	public void absorbEvidence(SingularContinuousVariable variable, double value) {
		potential.logNormalization +=
				potential.content.absorbEvidence(variable, value);
	}

	@Override
	public Message computeMessage(Message multiplier, Separator separator,
			Set<DiscreteVariable> retainingVariables) {
		// it should contain only the discrete variable of the separator,
		// so the retaining variables are ignored.

		Function function = potential.content.marginalize(separator.variable());
		Message message = new Message(function, potential.logNormalization);
		return multiplier == null ? message : message.times(multiplier);
	}

	@Override
	public void reset() {
		potential = null;
	}

	@Override
	public void combine(Potential other, double logNormalization) {
		if (other instanceof CGPotential) {
			combine((CGPotential) other, logNormalization);
		} else {
			combine(other.function(), logNormalization);
		}

		if (pivot) {
			normalize(Double.NaN);
		}
	}

	private void combine(CGPotential other, double logNormalization) {
		if (potential == null) {
			potential =
					new CliquePotential<CGPotential>(other.clone(),
							logNormalization);
		} else {
			potential.content.combine(other);
			potential.logNormalization += logNormalization;
		}
	}

	private void combine(Function other, double logNormalization) {
		if (potential == null) {
			CGPotential p = new CGPotential(jointVariable, discreteVariable);
			potential = new CliquePotential<CGPotential>(p, logNormalization);
		}

		potential.content.multiply(other.function());
		potential.logNormalization += logNormalization;
	}

	@Override
	public List<Variable> variables() {
		List<Variable> result = new ArrayList<Variable>();
		result.addAll(jointVariable.variables());
		result.add(discreteVariable);
		return result;
	}

	@Override
	public <T> T accept(Visitor<T> visitor) {
		return visitor.visit(this);
	}

	@Override
	public List<DiscreteVariable> discreteVariables() {
		return Collections.singletonList(discreteVariable);
	}

}
