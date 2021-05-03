package org.latlab.core.util;

import java.util.Set;

/**
 * Represents a continuous variable.
 * 
 * <p>
 * It forms a composite pattern with {@code SingularContinuousVariable} and
 * {@code JointContinuousVariable}.
 * 
 * @author leonard
 * 
 */
public abstract class ContinuousVariable extends Variable {

	/**
	 * Constructs a continuous variable with the given name.
	 * 
	 * @param name
	 *            name of the variable
	 */
	protected ContinuousVariable(String name) {
		super(name);
	}

	protected ContinuousVariable() {
	}

	/**
	 * Returns a set of underlying singular continuous variable.
	 * 
	 * @return a set of underlying singular continuous variable
	 */
	public abstract Set<SingularContinuousVariable> variables();
	
	public String toString() {
	    return String.format("%s(c)", getName());
	}
}
