/**
 * BeliefNode.java Copyright (C) 2006 Tao Chen, Kin Man Poon, Yi Wang, and Nevin
 * L. Zhang
 */
package org.latlab.core.model;

import org.latlab.core.graph.AbstractGraph;
import org.latlab.core.graph.AbstractNode;
import org.latlab.core.graph.DirectedNode;
import org.latlab.core.util.DiscreteVariable;
import org.latlab.core.util.Function;
import org.latlab.core.util.Potential;

import java.util.Random;

/**
 * This class provides an implementation for nodes in BNs.
 * 
 * @author Yi Wang
 * 
 */
public class DiscreteBeliefNode extends BeliefNode {

    /**
     * the variable attached to this node.
     */
    private DiscreteVariable _variable;

    /**
     * the conditional probability table (CPT) attached to this node.
     */
    private Function _cpt;

    /**
     * <p>
     * Constructs a node with the specified variable attached and the specified
     * graph to contain it. This node has the same name as the argument
     * variable.
     * </p>
     * 
     * <p>
     * <b>Note: Besides constructors of subclasses, only
     * <code>BayesNet.addNode(Variable)</code> is supposed call this method.
     * </b>
     * </p>
     * 
     * @param graph
     *            graph to contain this node.
     * @param variable
     *            variable to be attached to this node.
     * @see BayesNet#addNode(DiscreteVariable)
     */
    protected DiscreteBeliefNode(AbstractGraph graph, DiscreteVariable variable) {
        super(graph, variable);

        _variable = variable;

        // sets CPT as uniform distribution
        _cpt = Function.createUniformDistribution(variable);
    }

    /**
     * Returns the standard dimension, namely, the number of free parameters in
     * the CPT, of this node.
     * 
     * @author csct
     * @return the standard dimension of this node.
     */
    public final int computeDimension() {
        // let X and pi(X) be variable attached to this node and joint variable
        // attached to parents, respectively. the standard dimension equals
        // (|X|-1)*|pi(X)|.
        int dimension = _variable.getCardinality() - 1;

        dimension *=
            DiscreteVariable.getCardinality(getDiscreteParentVariables());

        return dimension;
    }

    /**
     * Suppose this is a Latent node in an HLCM. Then for the purpose of
     * satisfying regularity, the cardinality of this node is at most:
     * CardOfNeighbor1 x CardOFNeighbor2 x ... x CardOfNeighborN / The maximum
     * cardinality amony its N neighbors. This method compute this quantity for
     * check regularity.
     * 
     * <p>
     * If any of its neighbors is a continuous node, then this regularity
     * definition does not hold, and it returns the largest value of an integer.
     * 
     * @return The maximum possible cardinality of this node in an HLCM
     */
    public final int computeMaxPossibleCardInHLCM() {

        int product = 1;
        int max = 1;

        // a visitor find the cardinality for discrete and continuous nodes
        Visitor<Integer> visitor =
            new Visitor<Integer>() {
                public Integer visit(DiscreteBeliefNode node) {
                    return node.getVariable().getCardinality();
                }

                public Integer visit(ContinuousBeliefNode node) {
                    return -1;
                }
            };

        for (AbstractNode neighbor : getNeighbors()) {
            int cardinality = ((BeliefNode) neighbor).accept(visitor);
            // if the neighbor is a continuous node
            if (cardinality < 0)
                return Integer.MAX_VALUE;

            product *= cardinality;
            max = Math.max(max, cardinality);
        }

        // the product has exceeded the largest possible value
        if (product < 0) {
            return Integer.MAX_VALUE;
        } else {
            return product / max;
        }
    }

    /**
     * Returns the CPT attached to this node. For the sake of efficiency, this
     * implementation returns the reference to a private field. Make sure you
     * understand this before using this method.
     * 
     * @return the CPT attached to this node.
     */
    public final Function potential() {
        return _cpt;
    }

    /**
     * Returns the variable attached to this node. For the sake of efficiency,
     * this implementation returns the reference to a private field. Make sure
     * you understand this before using this method.
     * 
     * @return the variable attached to this node.
     */
    public final DiscreteVariable getVariable() {
        return _variable;
    }

    /**
     * Returns <code>true</code> if the specified function can be a valid CPT of
     * this node. Here the meaning of <b>valid</b> is kind of partial since we
     * only check this function is a function of this node variable and the
     * variables in its parents. However <b>validity</b> do Not guarantee that
     * this function is a conditional probability table of
     * <code>this._variable</code>
     * 
     * @param function
     *            function whose validity as a CPT is to be tested.
     * @return <code>true</code> if the specified function can be a valid CPT of
     *         this node.
     */
    public final boolean isValidCpt(Function function) {
        // valid CPT must contain exactly variables in this family
        if (function.getDimension() != getInDegree() + 1) {
            return false;
        }

        if (!function.contains(_variable)) {
            return false;
        }

        for (DirectedNode parent : getParents()) {
            if (!function.contains(((DiscreteBeliefNode) parent).getVariable())) {
                return false;
            }
        }

        return true;
    }

    /**
     * <p>
     * Randomly sets the parameters of this node.
     * </p>
     * 
     * <p>
     * <b>Note: Only <code>BayesNet.randomlyParameterize()</code> and
     * <code>BayesNet.randomlyParameterize(java.util.Collection)</code> are
     * supposed to call this method.
     * </p>
     * 
     * <p>
     * Notes add by csct: We consider a more general situation of calling this
     * method, that is, when variables contained in "_ctp" mismatch with the
     * family Variables of the node, or maybe even the "_ctp" is just null. The
     * cases often occur when structure learning, e.g. in
     * HLCM.introduceState4Root().
     * </p>
     * 
     * @see BayesNet#randomlyParameterize()
     * @see BayesNet#randomlyParameterize(java.util.Collection)
     */
    protected final void randomlyParameterize() {
        _cpt.randomlyDistribute(_variable);
    }

    public void generateRandomParameters() {
        randomlyParameterize();
    }

    public void generateRandomParameters(Random random) {
        randomlyParameterize(random);
    }

    public final void randomlyParameterize(Random random) {
        _cpt.randomlyDistribute(_variable, random);
    }

    @Override
    public void setPotential(Potential potential) {
        setCpt((Function) potential);
    }

    /**
     * Replaces the CPT attached to this node. This implementation will check
     * whether this Function cpt is a funtion of this node and its parent nodes.
     * However, it is not guaranteed that cpt satisfies the probability
     * constraint. Therefore, when use this method, make sure Function cpt is in
     * the form of a conditional probability of <code>this._variable</code>
     * 
     * @param cpt
     *            new CPT to be attached to this node.
     */
    public final void setCpt(Function cpt) {
        if (!isValidCpt(cpt)) {
            System.out.println("Invalid CPT!");
        }
        // CPT must be valid
        assert isValidCpt(cpt);

        _cpt = cpt;

        // loglikelihoods expire
        ((BayesNet) _graph).expireLoglikelihoods();
    }

    /**
     * Reorders the states of the variable of this node. It affects also the
     * probability table of its child nodes.
     * 
     * @{code order[i] == x} means that the state originally at x should now be
     *        put at i.
     * @param order
     */
    public void reorderStates(int[] order) {
        _variable.reorderStates(order);

        _cpt.reorderStates(_variable, order);

        for (DirectedNode child : getChildren()) {
            ((BeliefNode) child).potential().reorderStates(_variable, order);
        }
    }

    @Override
    public <T> T accept(Visitor<T> visitor) {
        return visitor.visit(this);
    }
}
