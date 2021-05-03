package org.latlab.core.model;

import org.latlab.core.util.ContinuousVariable;
import org.latlab.core.util.DiscreteVariable;
import org.latlab.core.util.SingularContinuousVariable;
import org.latlab.core.util.Variable;

import java.util.Arrays;
import java.util.Collection;
import java.util.Set;

public class MixedVariableSet {
    private MixedVariableMap<Void, Void, Void> map;

    public MixedVariableSet() {
        map = new MixedVariableMap<Void, Void, Void>();
    }

    public MixedVariableSet(MixedVariableSet other) {
        map = new MixedVariableMap<Void, Void, Void>(other.map);
    }

    /**
     * 
     * @param initialCapacity
     *            initial capacity of the discrete map
     */
    public MixedVariableSet(int initialCapacity) {
        map = new MixedVariableMap<Void, Void, Void>(initialCapacity);
    }
    
    public MixedVariableSet(Collection<? extends Variable> variables) {
        this();
        addAll(variables);
    }

    public Set<SingularContinuousVariable> continuousSet() {
        return map.continuousMap().keySet();
    }

    public Set<DiscreteVariable> discreteSet() {
        return map.discreteMap().keySet();
    }

    public boolean contains(ContinuousVariable variable) {
        return map.containsKey(variable);
    }

    public boolean contains(SingularContinuousVariable variable) {
        return map.containsKey(variable);
    }

    public boolean contains(DiscreteVariable variable) {
        return map.containsKey(variable);
    }
    
    public boolean contains(Variable variable) {
        return map.containsKey(variable);
    }
    
    public void addAll(Collection<? extends Variable> variables) {
        for (Variable variable : variables) {
            add(variable);
        }   
    }

    public void add(Variable variable) {
        map.put(variable, null);
    }

    public void add(ContinuousVariable variable) {
        map.put(variable, null);
    }

    public void add(SingularContinuousVariable variable) {
        map.put(variable, null);
    }

    public void add(DiscreteVariable variable) {
        map.put(variable, null);
    }

    public void clear() {
        map.clear();
    }
    
    public int size() {
        return map.size();
    }

    public String toString() {
        return Arrays.asList((Object) continuousSet(), (Object) discreteSet())
            .toString();
    }

    /**
     * Finds a variable by name. It supports only singular variable (continuous
     * or discrete).
     * 
     * @param name
     *            name of the variable
     * @return variable with the given name, or {@code null} if not found
     */
    public Variable findVariableByName(String name) {
        return map.findVariableByName(name);
    }
}
