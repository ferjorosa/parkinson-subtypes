package org.latlab.core.model;

import org.latlab.core.graph.AbstractNode;
import org.latlab.core.graph.DirectedNode;
import org.latlab.core.graph.Edge;
import org.latlab.core.graph.search.PathFindingVisitor;
import org.latlab.core.util.Algorithm;
import org.latlab.core.util.Caster;
import org.latlab.core.util.DiscreteVariable;
import org.latlab.core.util.Function;

import java.util.List;

/**
 * Provides some functions for manipulating a Bayesian network model.
 * 
 * @author leonard
 * 
 */
public class ModelManipulator {
    /**
     * Replaces the old variable with the new variable.
     * 
     * @param model
     *            model in which the variable is replaced
     * @param oldVariable
     *            old variable
     * @param newVariable
     *            new variable
     */
    public static void replace(
        BayesNet model, DiscreteVariable oldVariable,
        DiscreteVariable newVariable) {
        DiscreteBeliefNode oldNode = model.getNode(oldVariable);
        DiscreteBeliefNode newNode = model.addNode(newVariable);

        for (DirectedNode parent : oldNode.getParents()) {
            model.addEdge(newNode, parent);
        }

        for (DirectedNode child : oldNode.getChildren()) {
            model.addEdge(child, newNode);
        }

        model.removeNode(oldNode);
    }

    /**
     * <p>
     * Root-walks in the given Bayesian network by step.
     * 
     * <p>
     * It assumes that the {@code model} has a tree structure. The {@code root}
     * is the root of the {@code model}. {@code newRoot} is the new root of the
     * model, and can only be a neighbor of the original root. This operation
     * reverses the direction of the edge between {@code root} and {@code
     * newRoot} and updates the prior probability on the new root and the
     * conditional probability of original root given the new root.
     * 
     * @param model
     *            a tree-structured Bayesian network
     * @param root
     *            original root
     * @param newRoot
     *            new root
     */
    public static void rootWalkByStep(
        BayesNet model, DiscreteBeliefNode root, DiscreteBeliefNode newRoot) {
        assert root.getParents().size() == 0;
        assert newRoot.getParent() == root;

        // assume the old root and new root are X and Y respectively,
        // we need to compute P(X|Y) and P(Y)

        // conditional probability P(X|Y) is given by normalizing
        // the joint probability P(Y|X)P(X) by Y,
        // and the prior probability is given by summing out X
        // from the joint probability
        Function joint = newRoot.potential().times(root.potential());

        Function prior = joint.sumOut(root.getVariable());
        joint.normalize(root.getVariable());

        // reverse the edge between root and newRoot
        Edge oldEdge = newRoot.getParentEdges().iterator().next();
        model.removeEdge(oldEdge);

        model.addEdge(root, newRoot);
        newRoot.setCpt(prior);
        root.setCpt(joint); // the joint becomes conditional after normalization
    }

    /**
     * Changes the root in given model and returns the new model. This assumes
     * the given model is a tree-structured. It updates the structure of the
     * given model and the probability tables.
     * 
     * @param model
     *            model to change root
     * @param newRoot
     *            variable of the new root
     */
    public static void changeRoot(BayesNet model, DiscreteVariable newRoot) {
        List<DiscreteBeliefNode> roots =
            Algorithm.filter(
                model.getNodes(), new Caster<DiscreteBeliefNode>(),
                DirectedNode.ROOT_PREDICATE);

        if (roots.size() != 1) {
            throw new IllegalArgumentException(
                "The given Bayesian network does not have a tree structure");
        }

        List<AbstractNode> path =
            PathFindingVisitor.findPath(model, roots.get(0), model
                .getNode(newRoot));

        for (int i = 0; i < path.size() - 1; i++) {
            rootWalkByStep(
                model, (DiscreteBeliefNode) path.get(i),
                (DiscreteBeliefNode) path.get(i + 1));
        }
    }
}
