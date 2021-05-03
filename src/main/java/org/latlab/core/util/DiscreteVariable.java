/**
 * Variable.java Copyright (C) 2006 Tao Chen, Kin Man Poon, Yi Wang, and Nevin
 * L. Zhang
 */
package org.latlab.core.util;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * <p>
 * This class provides an implementation for nominal variables. Although the
 * states of a nominal variable is not ordered, we index them by 0, 1, ...,
 * (cardinality - 1) in this implementation.
 * </p>
 * 
 * <p>
 * Note: This class has a natural ordering that is inconsistent with equals.
 * </p>
 * 
 * @author Yi Wang
 * 
 */
public class DiscreteVariable extends Variable {

    /**
     * the common prefix of default names of states.
     */
    private final static String STATE_PREFIX = "state";

    /**
     * Returns the list of default names of states for the specified
     * cardinality. If you have no preference on names of states of the next
     * variable instance, use this method to obtain the list of default names.
     * 
     * @param cardinality
     *            number of states.
     * @return the list of default names of states for the specified
     *         cardinality.
     */
    private final static ArrayList<String> createDefaultStates(int cardinality) {

        assert cardinality > 0;

        ArrayList<String> states = new ArrayList<String>();

        for (int i = 0; i < cardinality; i++) {
            states.add(STATE_PREFIX + i);
        }

        return states;
    }

    /**
     * the list of states of this variable.
     */
    private List<String> _states;

    /**
     * Constructs a variable with the specified name and the specified list of
     * states.
     * 
     * @param name
     *            name of the variable to be created.
     * @param states
     *            list of states of the variable to be created.
     */
    public DiscreteVariable(String name, List<String> states) {
        super(name);

        // states cannot be empty
        assert !states.isEmpty();

        _states = states;
    }

    /**
     * Construct a variable with default nama and given number of states.
     * 
     * @author csct
     * @param cardinality
     *            The cardinality of this Variable.
     */
    public DiscreteVariable(int cardinality) {
        _states = createDefaultStates(cardinality);
    }

    /**
     * Returns <code>true</code> if the specified object is equals to this
     * variable. An object is equals to this variable if (1) it is a variable;
     * and (2) it has the same name and the same list of states as this
     * variable.
     * 
     * @return <code>true</code> if the specified object is equals to this
     *         variable.
     */
    public final boolean equals(Object object) {
        // tests identity
        if (this == object) {
            return true;
        } else if (object instanceof DiscreteVariable) {
            DiscreteVariable variable = (DiscreteVariable) object;
            return (getName().equals(variable.getName()) && _states
                .equals(variable._states));
        } else
            return false;
    }

    /**
     * Returns the cardinality of this variable. The cardinality of a nominal
     * variable equals the number of states that this variable can take.
     * 
     * @return the cardinality of this variable.
     */
    public final int getCardinality() {
        return _states.size();
    }

    /**
     * Returns the list of states of this variable.
     * 
     * @return the list of states of this variable.
     */
    public final List<String> getStates() {
        return _states;
    }

    /**
     * Returns the index of the specified state in the domain of this variable.
     * 
     * @param state
     *            state whose index is to be returned.
     * @return the index of the specified state in the domain of this variable.
     */
    public final int indexOf(String state) {
        return _states.indexOf(state);
    }

    /**
     * Returns <code>true</code> if the specified integer is a valid state index
     * in the domain of this variable.
     * 
     * @param state
     *            index of state whose validity is to be tested.
     * @return <code>true</code> if the specified integer is a valid state index
     *         in the domain of this variable.
     */
    public final boolean isValid(int state) {
        return (state >= 0 && state < getCardinality());
    }

    @Override
    public String toString() {
        return String.format("%s(%d)", getName(), getCardinality());
    }

    // /**
    // * Returns a string representation of this Variable. This implementation
    // * returns <code>toString(0)</code>.
    // *
    // * @return a string representation of this Variable.
    // * @see #toString(int)
    // */
    // public final String toString() {
    // return toString(0);
    // }

    /**
     * Returns a string representation of this Variable. The string
     * representation will be indented by the specified amount.
     * 
     * @param amount
     *            amount by which the string representation is to be indented.
     * @return a string representation of this Variable.
     */
    public final String toString(int amount) {
        // amount must be non-negative
        assert amount >= 0;

        // prepares white space for indent
        StringBuffer whiteSpace = new StringBuffer();
        for (int i = 0; i < amount; i++) {
            whiteSpace.append('\t');
        }

        // builds string representation
        StringBuffer stringBuffer = new StringBuffer();

        stringBuffer.append(whiteSpace);
        stringBuffer.append("Variable ");
        stringBuffer.append("\"" + getName() + "(");
        stringBuffer.append(_states.size() + ")\":");

        for (int i = 0; i < _states.size(); i++) {
            stringBuffer.append(" " + _states.get(i));
        }
        stringBuffer.append(whiteSpace);
        stringBuffer.append("\n");
        return stringBuffer.toString();
    }

    /**
     * Reorder the states according to the given order.
     * 
     * @param order
     *            shows the desired order, the array index is the position of
     *            the states, of which the index are held as items.
     */
    public void reorderStates(int[] order) {
        assert order.length == _states.size();
        List<String> clone = new ArrayList<String>(_states);
        for (int i = 0; i < _states.size(); i++) {
            _states.set(i, clone.get(order[i]));
        }
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

    /**
     * Returns the cardinality of a list of discrete variables, i.e., the number
     * of possible combinations of their states.
     * 
     * @param variables
     *            list of discrete variables
     * @return cardinality of the discrete variables
     */
    public static int getCardinality(Collection<DiscreteVariable> variables) {
        int cardinality = 1;
        for (DiscreteVariable variable : variables) 
            cardinality *= variable.getCardinality();
        
        return cardinality;
    }
}
