package org.latlab.core.reasoner;

import org.latlab.core.model.MixedVariableMap;
import org.latlab.core.util.DataSet;
import org.latlab.core.util.DiscreteVariable;
import org.latlab.core.util.SingularContinuousVariable;
import org.latlab.core.util.Variable;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 
 * <p>
 * <b>Caution</b>: If some evidence is set to a value that is impossible, i.e.
 * P(E) = 0, the propagation will lead to potentials with parameters equal to
 * {@code Double.NaN}.
 * 
 * @author leonard
 * 
 */
public class Evidences {
	private final MixedVariableMap<Number, Double, Integer> entries;

	public Evidences() {
		entries = new MixedVariableMap<Number, Double, Integer>();
	}

	private Evidences(Evidences other) {
		entries = new MixedVariableMap<Number, Double, Integer>(other.entries);
	}

	/**
	 * Projects this set of evidences on the given variables. Only the evidences
	 * to the given variables are included in the projected evidences.
	 * 
	 * @param variables
	 *            on which evidences are projected
	 * @return projected evidences
	 */
	public Evidences project(Set<? extends Variable> variables) {
		Evidences projected = new Evidences();
		for (Map.Entry<SingularContinuousVariable, Double> entry : entries.continuousMap().entrySet()) {
			if (variables.contains(entry.getKey())) {
				projected.add(entry.getKey(), entry.getValue());
			}
		}

		for (Map.Entry<DiscreteVariable, Integer> entry : entries.discreteMap().entrySet()) {
			if (variables.contains(entry.getKey())) {
				projected.add(entry.getKey(), entry.getValue());
			}
		}

		return projected;
	}

	public void add(SingularContinuousVariable variable, double value) {
		// if the value is a missing value, ignore it
		if (Double.isNaN(value)) {
			entries.remove(variable);
		} else {
			entries.put(variable, value);
		}
	}

	public void add(DiscreteVariable variable, int state) {
		assert variable.isValid(state);

		if (state == DataSet.MISSING_VALUE) {
			entries.remove(variable);
		} else {
			entries.put(variable, state);
		}
	}

	public void add(List<DiscreteVariable> variables, List<Integer> states) {
		for (int i = 0; i < variables.size(); i++) {
			add(variables.get(i), states.get(i));
		}
	}

	public void clear() {
		entries.clear();
	}

	public Map<SingularContinuousVariable, Double> continuous() {
		return Collections.unmodifiableMap(entries.continuousMap());
	}

	public Map<DiscreteVariable, Integer> discrete() {
		return Collections.unmodifiableMap(entries.discreteMap());
	}

	@Override
	public String toString() {
		return entries.toString();
	}

	/**
	 * @return
	 */
	public Evidences copy() {
		return new Evidences(this);
	}
}
