package eu.amidst.extension.util.rfmetric;

import java.util.List;
import java.util.Objects;

public class RFPartition {

    RFNode parent;

    List<RFNode> nodes;

    RFPartition(RFNode parent, List<RFNode> nodes) {
        this.parent = parent;
        this.nodes = nodes;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        RFPartition that = (RFPartition) o;
        if (!Objects.equals(parent, that.parent))
            return false;
        return nodes.containsAll(that.nodes) &&
                nodes.size() == that.nodes.size();
    }

    @Override
    public int hashCode() {

        return Objects.hash(parent, nodes);
    }
}
