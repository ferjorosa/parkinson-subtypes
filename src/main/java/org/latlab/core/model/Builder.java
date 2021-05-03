/**
 * 
 */
package org.latlab.core.model;

import org.latlab.core.graph.AbstractNode;
import org.latlab.core.graph.DirectedNode;
import org.latlab.core.graph.Edge;
import org.latlab.core.util.DiscreteVariable;
import org.latlab.core.util.JointContinuousVariable;
import org.latlab.core.util.SingularContinuousVariable;
import org.latlab.core.util.Variable;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * @author leonard
 * 
 */
public class Builder {

    /**
     * Constructs a local independence model with the given observed {@code
     * variables} as leaf nodes and the given class variable as parent to each
     * of the leaf nodes. leaf nodes.
     * 
     * @param <T>
     *            type of the model
     * @param model
     *            model to build on
     * @param observedVariables
     *            observed variables
     * @return a local independence model
     */
    public static <T extends BayesNet> T buildLocalIndependenceModel(
        T model, Collection<? extends Variable> observedVariables) {
        return buildNaiveBayesModel(
            model, new DiscreteVariable(2), observedVariables);
    }

    /**
     * Constructs a local independence model with the given observed {@code
     * variables} as leaf nodes and the given class variable as parent to each
     * of the leaf nodes. leaf nodes.
     * 
     * @param <T>
     *            type of the model
     * @param model
     *            model to build on
     * @param states
     *            number of states in the latent variable
     * @param observedVariables
     *            observed variables
     * @return a local independence model
     */
    public static <T extends BayesNet> T buildLocalIndependenceModel(
        T model, int states, Collection<? extends Variable> observedVariables) {
        return buildNaiveBayesModel(
            model, new DiscreteVariable(states), observedVariables);
    }

    /**
     * Constructs a Naive Bayes model with the given observed {@code variables}
     * as leaf nodes and the given class variable as parent to each of the leaf
     * nodes.
     * 
     * @param <T>
     *            type of the model
     * @param model
     *            model to build on
     * @param classVariable
     *            class variable
     * @param observedVariables
     *            observed variables
     * @return a local independence model
     */
    public static <T extends BayesNet> T buildNaiveBayesModel(
        T model,
        DiscreteVariable classVariable,
        Collection<? extends Variable> observedVariables) {

        List<BeliefNode> observedNodes =
            new ArrayList<BeliefNode>(observedVariables.size());

        for (Variable variable : observedVariables) {
            observedNodes.add(model.addNode(variable));
        }

        BeliefNode root = model.addNode(classVariable);

        for (BeliefNode leaf : observedNodes) {
            model.addEdge(leaf, root);
        }

        return model;
    }

    public static <T extends BayesNet> T buildMixtureModel(
        T model, int components,
        Collection<SingularContinuousVariable> variables) {
        BeliefNode root = model.addNode(new DiscreteVariable(components));
        BeliefNode leaf = model.addNode(new JointContinuousVariable(variables));
        model.addEdge(leaf, root);
        return model;
    }

    public static <T extends BayesNet> T buildMixedMixtureModel(
        final T model, int components, Collection<Variable> variables) {
        return buildMixedMixtureModel(
            model, new DiscreteVariable(components), variables);
    }

    public static <T extends BayesNet> T buildMixedMixtureModel(
        final T model, DiscreteVariable rootVariable,
        Collection<Variable> variables) {
        final BeliefNode root = model.addNode(rootVariable);

        final List<SingularContinuousVariable> continuousVariables =
            new ArrayList<SingularContinuousVariable>(variables.size());

        for (Variable variable : variables) {
            variable.accept(new Variable.Visitor<Void>() {

                @Override
                public Void visit(DiscreteVariable variable) {
                    BeliefNode node = model.addNode(variable);
                    model.addEdge(node, root);
                    return null;
                }

                @Override
                public Void visit(JointContinuousVariable variable) {
                    continuousVariables.addAll(variable.variables());
                    return null;
                }

                @Override
                public Void visit(SingularContinuousVariable variable) {
                    continuousVariables.add(variable);
                    return null;
                }

            });
        }

        BeliefNode node =
            model.addNode(new JointContinuousVariable(continuousVariables));
        model.addEdge(node, root);

        return model;
    }

    /**
     * Removes the original parent of the given variables and connects them to
     * the given parent. The new parent is then connected as a child to the
     * original parent. It assumes the variables belong to the same original
     * parent. Also, the parent variable has not been added to the network.
     * 
     * @param <T>
     * @param model
     * @param observedVariables
     * @return
     */
    public static void introduceLatentNode(
        BayesNet model, DiscreteVariable parent,
        Collection<? extends Variable> variables) {

        AbstractNode original = null;
        BeliefNode parentNode = model.addNode(parent);

        for (Variable variable : variables) {
            BeliefNode node = model.getNode(variable);
            for (Edge edge : new ArrayList<Edge>(node.getParentEdges())) {
                original = edge.getTail();
                model.removeEdge(edge);
            }

            model.addEdge(node, parentNode);
        }

        assert original != null;
        model.addEdge(parentNode, original);
    }

    /**
     * Replaces the old variable with the new variable and returns the new node
     * with the new variable.
     * 
     * @param model
     * @param oldVariable
     * @param newVariable
     * @return
     */
    public static DiscreteBeliefNode replaceVariable(
        BayesNet model, DiscreteVariable oldVariable,
        DiscreteVariable newVariable) {

        DiscreteBeliefNode oldNode = model.getNode(oldVariable);
        DiscreteBeliefNode newNode = model.addNode(newVariable);

        // It should have only one parent, but we play safe here in case of
        // future extension. It has to make a copy here because modifying
        // the model may have changed these sets.
        List<DirectedNode> parents =
            new ArrayList<DirectedNode>(oldNode.getParents());
        List<DirectedNode> children =
            new ArrayList<DirectedNode>(oldNode.getChildren());

        model.removeNode(oldNode);

        // add the new node with one more state and connect it back to the
        // old neighbors
        for (DirectedNode parent : parents)
            model.addEdge(newNode, parent);
        for (DirectedNode child : children)
            model.addEdge(child, newNode);

        return newNode;
    }
}
