package org.latlab.core.util;

public interface Potential {
    /**
     * Adds a variable and returns the modified potential. This potential may be
     * used as the returned value.
     * 
     * @param variable
     *            variable to add
     * @return the modified potential
     */
    public Potential addParentVariable(Variable variable);

    /**
     * Removes a variable and returns the modified potential. This potential may
     * be used as the returned value.
     * 
     * @param variable
     *            variable to remove
     * @return the modified potential
     */
    public Potential removeParentVariable(Variable variable);

    /**
     * Normalizes this potential and returns the normalization constant. If the
     * normalization constant is given, it is used. Otherwise if it is {@code
     * Double.NaN}, the constant is computed.
     * 
     * @param constant
     *            normalization constant or {@code Double.NaN}
     * @return normalization constant used
     */
    public double normalize(double constant);

    /**
     * It times a indicator function with the only {@code variable} and value 1
     * at the {@state}.
     * 
     * @param variable
     *            variable of the indicator function
     * @param state
     *            state which has value 1, and other unspecified states 0
     */
    public void timesIndicator(DiscreteVariable variable, int state);

    /**
     * Returns a copy of this potential. The copy and the this potential shares
     * the same instances of variables.
     * 
     * @return a copy of this potential
     */
    public Potential clone();

    /**
     * Returns a function holding only the discrete part.
     * 
     * @return function of the discrete part
     */
    public Function function();

    /**
     * Marginalizes this function to have only the given {@code variable}.
     * 
     * @param variable
     *            variable left in the marginalized function
     * @return function marginalized to the given {@code variable}
     */
    public Function marginalize(DiscreteVariable variable);
    
    /**
     * Reorders the states according to the given order by adjusting the
     * positions of the probability entries in this function.
     * 
     * @param variable
     *            variable states of which is being reordered
     * @param order
     *            the new state order
     */
    public void reorderStates(DiscreteVariable variable, int[] order);
}
