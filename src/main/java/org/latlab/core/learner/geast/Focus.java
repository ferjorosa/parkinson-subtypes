package org.latlab.core.learner.geast;

import org.latlab.core.model.MixedVariableSet;
import org.latlab.core.util.Variable;

import java.util.Collection;

/**
 * Holds the variables under focus in a propagation.
 * 
 * <p>
 * A variable under focus usually means that the probability table of that
 * variable needs to be updated. Therefore, only the clique corresponding to
 * that probability table needs to be set focus, but not all cliques containing
 * that variable.
 * 
 * @author leonard
 * 
 */
public class Focus extends MixedVariableSet {

	/**
	 * Default constructor.
	 */
	public Focus() {
	}

	/**
	 * Constructs the focus with the given variables.
	 * 
	 * @param variables
	 *            initial variables under focus
	 */
	public Focus(Collection<? extends Variable> variables) {
		super(variables);
	}

	/**
	 * Returns the pivot in a clique tree for this focus.
	 * 
	 * @return the pivot
	 */
	public Variable pivot() {
		if (discreteSet().size() > 0) {
			return discreteSet().iterator().next();
		} else if (continuousSet().size() > 0) {
			return continuousSet().iterator().next();
		} else {
			return null;
		}
	}

}
