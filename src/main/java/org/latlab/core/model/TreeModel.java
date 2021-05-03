package org.latlab.core.model;

import org.latlab.core.graph.AbstractNode;
import org.latlab.core.graph.DirectedNode;
import org.latlab.core.util.Algorithm;
import org.latlab.core.util.Caster;
import org.latlab.core.util.Variable;

import java.util.ArrayList;
import java.util.List;

/**
 * Represents a tree-structured Bayesian networks.
 * 
 * <p>
 * In the current implementation, it does not check for the validity of this
 * network during the model modification operations.
 * 
 * @author leonard
 * 
 */
public class TreeModel extends BayesNet {

    public TreeModel(String name) {
        super(name);
    }

    public TreeModel() {

    }

    protected TreeModel(TreeModel other) {
        super(other);
    }

    /**
     * Get root node of this tree model, or {@code null} if this model does not
     * contain any node.
     * 
     * @return Root node.
     */
    public BeliefNode getRoot() {
        List<AbstractNode> allNodes = getNodes();

        // return null for an empty model.
        if (allNodes.isEmpty())
            return null;

        DirectedNode node = (DirectedNode) allNodes.get(0);

        while (!node.isRoot()) {
            node = node.getParent();
        }
        return (BeliefNode) node;
    }

    public List<DiscreteBeliefNode> getInternalNodes() {
        return Algorithm.filter(
            _variables.discreteMap().values(), DirectedNode.INTERNAL_PREDICATE);
    }

    public List<BeliefNode> getLeafNodes() {
        return Algorithm.filter(
            getNodes(), new Caster<BeliefNode>(), DirectedNode.LEAF_PREDICATE);
    }
    
    public List<Variable> getLeafVariables() {
        return Algorithm.convert(getLeafNodes(), new VariableExtractor());
    }
    
//    public List<Variable> getInternalVariables() {
//    	return Algorithm.convert(getInternalNodes(), new VariableExtractor());
//    }
//    
    public List<Variable> getLeafSingularVariables() {
        List<BeliefNode> nodes = getLeafNodes();
        final List<Variable> variables = new ArrayList<Variable>(nodes.size());

        for (BeliefNode node : nodes) {
            node.accept(new BeliefNode.Visitor<Void>() {

                public Void visit(DiscreteBeliefNode node) {
                    variables.add(node.getVariable());
                    return null;
                }

                public Void visit(ContinuousBeliefNode node) {
                    variables.addAll(node.getVariable().variables());
                    return null;
                }
                
            });
        }
        
        return variables;
    }
}
