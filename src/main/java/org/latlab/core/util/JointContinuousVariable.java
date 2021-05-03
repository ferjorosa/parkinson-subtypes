package org.latlab.core.util;

import java.util.*;

/**
 * Represents a joint continuous variable, which can possibly be a
 * multidimensional variable. It holds a list of singular continuous variable,
 * and should contain at least one such variable.
 * 
 * @author leonard
 * 
 */
public class JointContinuousVariable extends ContinuousVariable {
    /**
     * Holds the variables that are joint together. This set has to be sorted so
     * that the same variables contained by different joint variables have the
     * same ordering.
     */
    private SortedSet<SingularContinuousVariable> variables;

    /**
     * Constructs a joint continuous variable from a singular continuous
     * variable.
     */
    public JointContinuousVariable(SingularContinuousVariable variable) {
        this(Collections.singletonList(variable));
    }

    public JointContinuousVariable(
        Collection<SingularContinuousVariable> variables) {
        this.variables = new TreeSet<SingularContinuousVariable>(variables);
        setName(constructName());
    }

    public JointContinuousVariable(
        JointContinuousVariable variable1, JointContinuousVariable variable2) {
        variables = new TreeSet<SingularContinuousVariable>();
        variables.addAll(variable1.variables());
        variables.addAll(variable2.variables());
        setName(constructName());
    }

    private JointContinuousVariable(
        TreeSet<SingularContinuousVariable> variables) {
        this.variables = variables;
        setName(constructName());
    }

    /**
     * Attaches the given set of variables to a new joint variable and returns
     * this new variable.
     * <p>
     * Note that changing the content of the given affects the new variable and
     * may have unexpected effect.
     * 
     * @param variables
     *            variables attached to the new joint variable
     * @return the new variable
     */
    public static JointContinuousVariable attach(
        TreeSet<SingularContinuousVariable> variables) {
        return new JointContinuousVariable(variables);
    }

    /**
     * Returns the variables forming this joint variable. The returned variables
     * cannot be modified.
     * 
     * @return variables forming this joint variable
     */
    public Set<SingularContinuousVariable> variables() {
        return Collections.unmodifiableSortedSet(variables);
    }

    // TODO: may consider not able to modify a joint variable, but only create a
    // new joint variable
    // /**
    // * Adds new variables to this joint variable.
    // *
    // * @param variables
    // * variables to add
    // */
    // public void add(Set<SingularContinuousVariable> variables) {
    // variables.addAll(variables);
    // setName(constructName());
    // }

    // TODO: may consider not able to modify a joint variable, but only create a
    // new joint variable
    /**
     * Removes a variables from this joint variable.
     * 
     * @param variables
     *            variables to remove
     */
    // public void remove(Collection<SingularContinuousVariable> variables) {
    // variables.removeAll(variables);
    // setName(constructName());
    //
    // if (variables.size() < 1) {
    // throw new IllegalStateException(
    // "The joint variable has to contain at least one variable.");
    // }
    // }
    /**
     * Constructs a name for this variable, which is a comma separated list of
     * the names of its containing variables.
     * 
     * @return name of this variable
     */
    private String constructName() {
        return getName(variables(), ", ");
    }

    /**
     * The accept function of the visitor pattern.
     * 
     * @param visitor
     *            visits this instance
     */
    public <T> T accept(Visitor<T> visitor) {
        return visitor.visit(this);
    }

    @Override
    public boolean equals(Object o) {
        if (o == this)
            return true;
        if (o instanceof ContinuousVariable) {
            return variables().equals(((ContinuousVariable) o).variables());
        } else
            return false;
    }
    
    @Override
    public int hashCode() {
        return variables().hashCode();
    }
}
