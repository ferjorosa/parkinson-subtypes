package eu.amidst.extension.util.rfmetric;

import java.util.List;
import java.util.Objects;

public class RFNode {

    enum VarType { MANIFEST, LATENT, NONE }

    enum VarSpace { DISCRETE, CONTINUOUS, NONE }

    String name;

    VarType varType;

    VarSpace varSpace;

    List<RFNode> children;

    int cardinality;

    RFNode(String name, VarType varType, VarSpace varSpace, List<RFNode> children, int cardinality) {
        this.name = name;
        this.varType = varType;
        this.varSpace = varSpace;
        this.children = children;
        this.cardinality = cardinality;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        RFNode rfNode = (RFNode) o;
        return Objects.equals(name, rfNode.name) &&
                varType == rfNode.varType &&
                varSpace == rfNode.varSpace &&
                cardinality == rfNode.cardinality;
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, varType, varSpace, cardinality);
    }

    @Override
    public String toString() {

        String s = this.name;
        if(this.varType == VarType.LATENT && this.varSpace == VarSpace.DISCRETE)
            s+= "("+this.cardinality+")";

        if (!children.isEmpty()) {
            s += " {";
            for (int i = 0; i < children.size(); i++) {
                RFNode child = children.get(i);
                s += child.name;
                if(child.varType == VarType.LATENT && child.varSpace == VarSpace.DISCRETE)
                    s+= "("+child.cardinality+")";
                if (i < (children.size() - 1))
                    s += ", ";
            }
            s += "}";
        }
        return s;
    }
}