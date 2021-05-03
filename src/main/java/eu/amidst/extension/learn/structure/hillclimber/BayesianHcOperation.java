package eu.amidst.extension.learn.structure.hillclimber;

import eu.amidst.core.variables.Variable;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;

public class BayesianHcOperation {

    public enum Type {
        ADD_ARC,
        REMOVE_ARC,
        REVERSE_ARC
    }

    private Variable fromVar;

    private Variable toVar;

    private double totalScore; // The combined score of all node scores

    private Type type;

    public BayesianHcOperation(Variable fromVar,
                               Variable toVar,
                               double totalScore,
                               Type type) {
        this.fromVar = fromVar;
        this.toVar = toVar;
        this.totalScore = totalScore;
        this.type = type;
    }

    public Variable getFromVar() {
        return fromVar;
    }

    public Variable getToVar() {
        return toVar;
    }

    public double getTotalScore() {
        return totalScore;
    }

    public Type getType() {
        return type;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;

        if (o == null || getClass() != o.getClass()) return false;

        BayesianHcOperation that = (BayesianHcOperation) o;

        return new EqualsBuilder()
                .append(fromVar, that.fromVar)
                .append(toVar, that.toVar)
                .append(type, that.type)
                .isEquals();
    }

    @Override
    public int hashCode() {
        return new HashCodeBuilder(17, 37)
                .append(fromVar)
                .append(toVar)
                .append(type)
                .toHashCode();
    }
}
