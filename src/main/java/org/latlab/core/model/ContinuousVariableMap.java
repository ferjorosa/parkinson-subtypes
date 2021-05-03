package org.latlab.core.model;

import org.latlab.core.util.ContinuousVariable;
import org.latlab.core.util.SingularContinuousVariable;

import java.util.HashMap;

/**
 * A map from continuous variable to another type of element. It handles the
 * difference between singular continuous variable and joint continuous
 * variable, so that a joint variable and its containing singular variables are
 * mapped to the same target.
 * 
 * @author leonard
 * 
 * @param <T>
 *            type of target element
 */
public class ContinuousVariableMap<T>
    extends HashMap<SingularContinuousVariable, T> {

    private static final long serialVersionUID = -4882169413636870399L;

    public ContinuousVariableMap() {}

    public ContinuousVariableMap(ContinuousVariableMap<T> other) {
        super(other);
    }

    public T put(ContinuousVariable variable, T value) {
        T result = null;
        for (SingularContinuousVariable singular : variable.variables()) {
            T last = put(singular, value);
            if (result == null)
                result = last;
        }

        return result;
    }

    public T get(SingularContinuousVariable variable) {
        return super.get(variable);
    }

    public T get(ContinuousVariable variable) {
        return super.get(variable.variables().iterator().next());
    }

    /**
     * Overrides the super class method to prevent a continuous variable to be
     * used directly as a key.
     * 
     * @deprecated This method should not be called usually, and is implemented
     *             only to prevent the pitfall from calling this
     *             unintentionally.
     */
    @Override
    @Deprecated
    public T get(Object key) {
        if (key instanceof ContinuousVariable)
            return get((ContinuousVariable) key);
        return super.get(key);
    }

    public T remove(SingularContinuousVariable variable) {
        return super.remove(variable);
    }

    public T remove(ContinuousVariable variable) {
        T result = null;
        for (SingularContinuousVariable singular : variable.variables()) {
            T last = remove(singular);
            if (result == null)
                result = last;
        }

        return result;
    }

    /**
     * Overrides the super class method to prevent a continuous variable to be
     * used directly as a key.
     * 
     * @deprecated This method should not be called usually, and is implemented
     *             only to prevent the pitfall from calling this
     *             unintentionally.
     */
    @Override
    @Deprecated
    public T remove(Object key) {
        if (key instanceof ContinuousVariable)
            return get((ContinuousVariable) key);
        return super.remove(key);
    }

    public boolean containsKey(SingularContinuousVariable variable) {
        return super.containsKey(variable);
    }

    public boolean containsKey(ContinuousVariable variable) {
        return keySet().containsAll(variable.variables());
    }

    /**
     * Overrides the super class method to prevent a continuous variable to be
     * used directly as a key.
     * 
     * @deprecated This method should not be called usually, and is implemented
     *             only to prevent the pitfall from calling this
     *             unintentionally.
     */
    @Override
    @Deprecated
    public boolean containsKey(Object key) {
        if (key instanceof SingularContinuousVariable) {
            return containsKey((SingularContinuousVariable) key);
        } else if (key instanceof ContinuousVariable) {
            return containsKey((ContinuousVariable) key);
        } else {
            return super.containsKey(key);
        }
    }
}
