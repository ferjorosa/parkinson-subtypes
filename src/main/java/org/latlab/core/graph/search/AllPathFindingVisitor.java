package org.latlab.core.graph.search;

import org.latlab.core.graph.AbstractNode;
import org.latlab.core.graph.Edge;

import java.util.HashMap;
import java.util.Map;

/**
 * Used to find the paths from all the nodes to a destination node in an
 * undirected graph. It also accepts a directed graph but ignores the direction
 * of the edges.
 * 
 * <p>
 * The first node discovered is considered to be the destination node. It also
 * resets the map when the first node is discovered. Therefore this visitor
 * cannot be used for nodes in a disconnected graph.
 * 
 * @author leonard
 * 
 */
public class AllPathFindingVisitor extends AbstractVisitor {

    /**
     * Maps from a source node to its next neighbor node that in turn leads to
     * the destination node.
     */
    private final Map<AbstractNode, AbstractNode> nextNodes;

    public AllPathFindingVisitor(int numberOfNodes) {
        nextNodes = new HashMap<AbstractNode, AbstractNode>(numberOfNodes);
    }

    public boolean discover(AbstractNode node, Edge edge) {
        if (edge == null) {
            nextNodes.clear();
        }

        if (nextNodes.containsKey(node))
            return false;

        if (edge == null) {
            nextNodes.put(node, null);
        } else {
            nextNodes.put(node, edge.getOpposite(node));
        }
        
        return true;
    }

    public void finish(AbstractNode node) {}
    
    public AbstractNode next(AbstractNode node) {
        return nextNodes.get(node);
    }
}
