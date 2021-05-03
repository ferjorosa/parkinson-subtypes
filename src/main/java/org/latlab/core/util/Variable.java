package org.latlab.core.util;

import org.latlab.core.util.Counter.Instance;

import java.util.Collection;

public abstract class Variable implements Comparable<Variable> {

    /**
     * A visitor class of the visitor pattern that visits the subclasses of the
     * {@code Variable} class.
     * 
     * @author leonard
     * 
     */
    public static class Visitor<T> {
        public T visit(SingularContinuousVariable variable) {
            return null;
        }

        public T visit(JointContinuousVariable variable) {
            return null;
        }

        public T visit(DiscreteVariable variable) {
            return null;
        }
    }

    private static Counter counter = new Counter("variable");

    /**
     * the index of this variable that indicates when it was created.
     */
    private int _index;

    /**
     * the name of this variable.
     */
    private String _name;

    /**
     * Constructs a variable with the given name, trimmed with the white spaces.
     * 
     * @param name
     *            name of the variable
     */
    protected Variable(String name) {
        _name = name.trim();

        // name cannot be blank
        assert _name.length() > 0;

        _index = counter.nextIndex();
        
        counter.encounterName(name);
    }

    /**
     * Constructs a variable by assigning it with a default name given by an
     * internal counter.
     */
    protected Variable() {
        Instance instance = counter.next();
        _name = instance.name;
        _index = instance.index;
    }

    /**
     * <p>
     * Compares this variable with the specified object for order.
     * </p>
     * 
     * <p>
     * Note: <code>compareTo(Object)</code> is inconsistent with
     * <code>equals(Object)</code>.
     * </p>
     * 
     * @param object
     *            the object to be compared.
     * @return a negative or a positive integer if this variable was created
     *         earlier than or later than the specified variable; zero if they
     *         refers to the same variable.
     */
    public int compareTo(Variable o) {
        return _index - o._index;
    }

    /**
     * Returns the name of this variable.
     * 
     * @return name of this variable
     */
    public String getName() {
        return _name;
    }

    /**
     * Updates the name of this variable.
     * 
     * <p>
     * Note: Only <code>BeliefNode.setName(String></code> is supposed to call
     * this method. Abusing this method may cause inconsistency between names of
     * a belief node and the variable attached to it.
     * </p>
     * 
     * @param name
     *            new name of this variable.
     */
    public void setName(String name) {
        name = name.trim();

        // name cannot be blank
        assert name.length() > 0;

        _name = name;
    }

    public abstract <T> T accept(Visitor<T> visitor);

    /**
     * Returns a string representation of a collection of variables, with a
     * delimiter between each of them.
     * 
     * @param variables
     * @param delimiter
     * @return
     */
    public static String getName(
        Collection<? extends Variable> variables, String delimiter) {
        boolean first = true;

        StringBuilder builder = new StringBuilder();
        for (Variable variable : variables) {
            if (!first) {
                builder.append(delimiter);
            }

            builder.append(variable.getName());
            first = false;
        }

        return builder.toString();
    }
}
