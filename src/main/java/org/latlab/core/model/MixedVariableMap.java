package org.latlab.core.model;

import org.latlab.core.util.*;

import java.util.*;

/**
 * Holds keys of continuous variables and discrete variables and their
 * corresponding values.
 * 
 * <p>
 * The two different types of variables may map to two different value types
 * with a common base type. This map keeps the type information of the values,
 * instead of storing them as the base type.
 * 
 * @author leonard
 * 
 * @param <BaseV>
 *            base type of the two value types
 * @param <ContinuousV>
 *            type of values for continuous variables
 * @param <DiscreteV>
 *            type of values for discrete variables
 */
public class MixedVariableMap<BaseV, ContinuousV extends BaseV, DiscreteV extends BaseV> {
    private ContinuousVariableMap<ContinuousV> continuousMap;
    private Map<DiscreteVariable, DiscreteV> discreteMap;

    public MixedVariableMap() {
        continuousMap = new ContinuousVariableMap<ContinuousV>();
        discreteMap = new HashMap<DiscreteVariable, DiscreteV>();
    }

    public MixedVariableMap(
        MixedVariableMap<BaseV, ContinuousV, DiscreteV> other) {
        continuousMap =
            new ContinuousVariableMap<ContinuousV>(other.continuousMap);
        discreteMap =
            new HashMap<DiscreteVariable, DiscreteV>(other.discreteMap);
    }

    /**
     * 
     * @param initialCapacity
     *            initial capacity of the discrete map
     */
    public MixedVariableMap(int initialCapacity) {
        continuousMap = new ContinuousVariableMap<ContinuousV>();
        discreteMap = new HashMap<DiscreteVariable, DiscreteV>(initialCapacity);
    }

    public BaseV get(Variable variable) {
        return variable.accept(new Variable.Visitor<BaseV>() {

            @Override
            public BaseV visit(DiscreteVariable variable) {
                return get(variable);
            }

            @Override
            public BaseV visit(JointContinuousVariable variable) {
                return get((ContinuousVariable) variable);
            }

            @Override
            public BaseV visit(SingularContinuousVariable variable) {
                return get(variable);
            }

        });
    }

    public ContinuousV get(ContinuousVariable variable) {
        return continuousMap.get(variable);
    }

    public ContinuousV get(SingularContinuousVariable variable) {
        return continuousMap.get(variable);
    }

    public DiscreteV get(DiscreteVariable variable) {
        return discreteMap.get(variable);
    }

    @SuppressWarnings("unchecked")
    public void put(Variable variable, final BaseV value) {
        variable.accept(new Variable.Visitor<Void>() {

            @Override
            public Void visit(DiscreteVariable variable) {
                put(variable, (DiscreteV) value);
                return null;
            }

            @Override
            public Void visit(JointContinuousVariable variable) {
                put((ContinuousVariable) variable, (ContinuousV) value);
                return null;
            }

            @Override
            public Void visit(SingularContinuousVariable variable) {
                put(variable, (ContinuousV) value);
                return null;
            }

        });
    }

    public void put(ContinuousVariable variable, ContinuousV value) {
        continuousMap.put(variable, value);
    }

    public void put(SingularContinuousVariable variable, ContinuousV value) {
        continuousMap.put(variable, value);
    }

    public void put(DiscreteVariable variable, DiscreteV value) {
        discreteMap.put(variable, value);
    }

    public Map<SingularContinuousVariable, ContinuousV> continuousMap() {
        return continuousMap;
    }

    public Map<DiscreteVariable, DiscreteV> discreteMap() {
        return discreteMap;
    }

    public ContinuousV remove(ContinuousVariable variable) {
        return continuousMap.remove(variable);
    }

    public ContinuousV remove(SingularContinuousVariable variable) {
        return continuousMap.remove(variable);
    }

    public DiscreteV remove(DiscreteVariable variable) {
        return discreteMap.remove(variable);
    }

    public BaseV remove(Variable variable) {
        return variable.accept(new Variable.Visitor<BaseV>() {

            @Override
            public BaseV visit(DiscreteVariable variable) {
                return remove(variable);
            }

            @Override
            public BaseV visit(JointContinuousVariable variable) {
                return remove((ContinuousVariable) variable);
            }

            @Override
            public BaseV visit(SingularContinuousVariable variable) {
                return remove(variable);
            }

        });
    }

    public boolean containsKey(ContinuousVariable variable) {
        return continuousMap.containsKey(variable);
    }

    public boolean containsKey(SingularContinuousVariable variable) {
        return continuousMap.containsKey(variable);
    }

    public boolean containsKey(DiscreteVariable variable) {
        return discreteMap.containsKey(variable);
    }

    public boolean containsKey(Variable variable) {
        return variable.accept(new Variable.Visitor<Boolean>() {

            @Override
            public Boolean visit(DiscreteVariable variable) {
                return containsKey(variable);
            }

            @Override
            public Boolean visit(JointContinuousVariable variable) {
                return containsKey(variable);
            }

            @Override
            public Boolean visit(SingularContinuousVariable variable) {
                return containsKey((ContinuousVariable) variable);
            }

        });
    }

    public void clear() {
        continuousMap.clear();
        discreteMap.clear();
    }

    public int size() {
        return continuousMap.size() + discreteMap.size();
    }

    public String toString() {
        return Arrays.asList((Object) continuousMap, (Object) discreteMap)
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
        Predicate<Variable> predicate = new VariableNamePredicate(name);

        Variable variable =
            Algorithm.linearSearch(continuousMap.keySet(), predicate);

        return variable == null ? Algorithm.linearSearch(
            discreteMap.keySet(), predicate) : variable;
    }

    /**
     * Returns an unmodifiable set of the keys contained.
     * 
     * @return an unmodifiable set of keys
     */
    public Set<Variable> keySet() {
        Set<Variable> keys = new HashSet<Variable>(size());
        keys.addAll(discreteMap().keySet());
        keys.addAll(continuousMap().keySet());

        return Collections.unmodifiableSet(keys);
    }

    public Collection<BaseV> values() {
        List<BaseV> values = new ArrayList<BaseV>(size());
        values.addAll(discreteMap().values());
        values.addAll(continuousMap().values());

        return values;
    }
}
