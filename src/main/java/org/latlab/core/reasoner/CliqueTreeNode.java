package org.latlab.core.reasoner;

import org.latlab.core.graph.AbstractGraph;
import org.latlab.core.graph.AbstractNode;
import org.latlab.core.graph.UndirectedNode;
import org.latlab.core.util.DiscreteVariable;
import org.latlab.core.util.Potential;
import org.latlab.core.util.Variable;

import java.util.List;

/**
 * Node in a clique tree. Can be a clique or a separator.
 * 
 * @author leonard
 * 
 */
public abstract class CliqueTreeNode extends UndirectedNode {

    public interface Visitor<T> {
        public T visit(Separator separator);

        public T visit(DiscreteClique clique);

        public T visit(MixedClique clique);
    }

    protected CliqueTreeNode(AbstractGraph graph, String name) {
        super(graph, name);
    }

    public abstract Potential potential();

    /**
     * Resets this clique tree node before a propagation.
     */
    public abstract void reset();

    /**
     * Returns a list of primary variables. The types of the variables are
     * either {@code DiscreteVariable} or {@code SingularContinuousVariable}.
     * 
     * @return list of primary variables
     */
    public abstract List<Variable> variables();

    public abstract List<DiscreteVariable> discreteVariables();

    public abstract <T> T accept(Visitor<T> visitor);

    @Override
    public String toString(int amount) {
        // amount must be non-negative
        assert amount >= 0;

        // prepares white space for indent
        StringBuffer whiteSpace = new StringBuffer();
        for (int i = 0; i < amount; i++) {
            whiteSpace.append("\t");
        }

        // builds string representation
        StringBuffer stringBuffer = new StringBuffer();

        stringBuffer.append(whiteSpace);
        stringBuffer.append(String.format("%s: %s {\n", getClass()
            .getSimpleName(), Variable.getName(variables(), ", ")));

        stringBuffer.append(whiteSpace);
        stringBuffer.append("\tneighbors = { ");

        for (AbstractNode neighbor : getNeighbors()) {
            stringBuffer.append("\"" + neighbor.getName() + "\" ");
        }

        stringBuffer.append("};\n");

        if (potential() != null) {
            stringBuffer.append(potential());
            stringBuffer.append("\n");
        } else
            stringBuffer.append("potential: nil\n");

        stringBuffer.append(whiteSpace);
        stringBuffer.append("};\n");

        return stringBuffer.toString();
    }
}
