package org.latlab.core.util;

/**
 * Evaluates a given object and gives a score for it.
 * 
 * @author leonard
 * 
 * @param <T>
 *            type of objects being evaluated
 */
public interface Evaluator<T> {
    /**
     * Returns the score of the given object.
     * 
     * @param o
     *            object being evaluated
     * @return score
     */
    public double evaluate(T o);
}
