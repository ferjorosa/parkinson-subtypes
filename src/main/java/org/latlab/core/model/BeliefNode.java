/**
 * BeliefNode.java Copyright (C) 2006 Tao Chen, Kin Man Poon, Yi Wang, and Nevin
 * L. Zhang
 */
package org.latlab.core.model;

import org.latlab.core.graph.AbstractGraph;
import org.latlab.core.graph.AbstractNode;
import org.latlab.core.graph.DirectedNode;
import org.latlab.core.graph.Edge;
import org.latlab.core.util.DiscreteVariable;
import org.latlab.core.util.Potential;
import org.latlab.core.util.Variable;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * This class provides an implementation for nodes in BNs.
 * 
 */
public abstract class BeliefNode extends DirectedNode {

    public static interface Visitor<T> {
        public T visit(DiscreteBeliefNode node);

        public T visit(ContinuousBeliefNode node);
    }

    /**
     * Constructs a belief node for {@code variable} in the {@code graph}.
     * 
     * @param graph
     *            in which the belief node is added to
     * @param variable
     *            variable of this node
     */
    protected BeliefNode(AbstractGraph graph, Variable variable) {
        super(graph, variable.getName());
    }

    /**
     * Returns the potential of this belief node.
     * 
     * @return potential of this belief node
     */
    public abstract Potential potential();

    /**
     * Uses the given {@code potential} to be the potential of this belief node.
     * It assumes that this potential is of the appropriate type.
     * 
     * @param potential
     *            potential to be used by this belief node
     */
    public abstract void setPotential(Potential potential);

    /**
     * Returns the variable of this node.
     * 
     * @return variable of this node
     */
    public abstract Variable getVariable();

    /**
     * <p>
     * Attachs the specified incoming edge to this node. This implementation
     * extends <code>DirectedNode.attachInEdge(Edge edge)</code> such that the
     * CPT of this node will be updated as well.
     * </p>
     * 
     * <p>
     * <b>Note: Only <code>BayesNet.addEdge(AbstractNode, AbstractNode)</code>
     * is supposed to call this method. </b>
     * </p>
     * 
     * @param edge
     *            incoming edge to be attached to this node.
     * @see BayesNet#addEdge(AbstractNode, AbstractNode)
     */
    protected void attachInEdge(Edge edge) {
        super.attachInEdge(edge);

        // new CPT should include variable attached to parent
        Variable parent = ((BeliefNode) edge.getTail()).getVariable();
        setPotential(potential().addParentVariable(parent));
    }

    /**
     * <p>
     * Detaches the specified incoming edge from this node. This implementation
     * extends <code>DirectedNode.detachInEdge(Edge)</code> such that the CPT
     * will be updated as well.
     * </p>
     * 
     * <p>
     * <b>Note: Only <code>BayesNet.removeEdge(Edge)</code> is supposed to call
     * this method. </b>
     * </p>
     * 
     * @param edge
     *            incoming edge to be detached from this node.
     * @see BayesNet#removeEdge(Edge)
     */
    protected void detachInEdge(Edge edge) {
        super.detachInEdge(edge);

        // new CPT should exclude variable attached to old parent
        Variable oldParent = ((BeliefNode) edge.getTail()).getVariable();
        setPotential(potential().removeParentVariable(oldParent));
    }

    /**
     * Returns the standard dimension, namely, the number of free parameters,
     * corresponding to this node.
     * 
     * @return the standard dimension of this node.
     */
    public abstract int computeDimension();

    /**
     * <p>
     * Replaces the name of this node.
     * </p>
     * 
     * <p>
     * <b>Note: Do NOT use this method if the BN that contains this node has
     * been cloned. Otherwise, in the copy, names of node and its attached
     * variable will be inconsistent in the copy. </b>
     * </p>
     * 
     * @param name
     *            new name of this node.
     */
    public final void setName(String name) {
        super.setName(name);

        // renames attached variable
        getVariable().setName(name);
    }

    /**
     * Returns a list containing the discrete variables of its parent nodes,
     * excluding any continuous variables.
     * 
     * @return list containing node variable and parent variables.
     */
    public List<DiscreteVariable> getDiscreteParentVariables() {
        final List<DiscreteVariable> list =
            new ArrayList<DiscreteVariable>(getParents().size());

        Collection<DirectedNode> parentNodes = getParents();
        for (DirectedNode parentNode : parentNodes) {
            BeliefNode beliefNode = (BeliefNode) parentNode;

            if (parentNode instanceof DiscreteBeliefNode)
                list.add((DiscreteVariable) beliefNode.getVariable());
        }

        return list;
    }

    /**
     * Expires the log likelihoods stored in the containing network.
     */
    protected void expireNetworkLogLikelihoods() {
        ((BayesNet) _graph).expireLoglikelihoods();
    }

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

        List<DiscreteVariable> parents = getDiscreteParentVariables();

        if (parents.size() > 0) {
            stringBuffer.append(String.format("P(%s| %s) {\n", getVariable()
                .toString(), Variable.getName(parents, ", ")));
        } else {
            stringBuffer.append(String.format("P(%s) {\n", getVariable()
                .toString()));
        }

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
