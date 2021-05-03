package org.latlab.core.graph.search;

import org.latlab.core.graph.AbstractGraph;
import org.latlab.core.graph.AbstractNode;
import org.latlab.core.graph.DirectedNode;
import org.latlab.core.graph.Edge;
import org.latlab.core.util.Algorithm;
import org.latlab.core.util.Caster;

import java.util.Collection;
import java.util.List;

/**
 * Performs a depth first search on a directed acyclic graph. A custom operation
 * based on the depth first search can be implemented by extending the {@code
 * Visitor} class.
 * 
 * @author leonard
 * 
 */
public class DepthFirstSearch {

    private final boolean ignoreDirection;

    private final AbstractGraph graph;

    /**
     * Constructor
     * 
     * @param graph
     *            a graph to search on
     */
    public DepthFirstSearch(AbstractGraph graph) {
        this(graph, false);
    }

    /**
     * Constructor.
     * 
     * @param graph
     *            graph to search on
     * @param ignoreDirection
     *            whether to ignore the direction of the edges, which is not
     *            meaningful to an undirected graph that has no direction anyway
     */
    public DepthFirstSearch(AbstractGraph graph, boolean ignoreDirection) {
        this.graph = graph;
        this.ignoreDirection = ignoreDirection;
    }

    /**
     * Performs the search. Repeated starts the search from all root nodes in
     * the graph.
     * 
     * @param visitor
     *            visitor for the nodes
     */
    public void perform(Visitor visitor) {
        List<DirectedNode> roots =
            Algorithm.filter(
                graph.getNodes(), new Caster<DirectedNode>(),
                DirectedNode.ROOT_PREDICATE);

        for (DirectedNode root : roots) {
            perform(root, visitor);
        }
    }

    /**
     * Performs the search. The search only visits those nodes connected to the
     * start node.
     * 
     * @param start
     *            the start node
     * @param visitor
     *            visitor for the nodes
     */
    public void perform(AbstractNode start, Visitor visitor) {
        // this graph must contain the argument node
        assert graph.containsNode(start);

        if (visitor.discover(start, null)) {
            transverseChildren(start, visitor);
            visitor.finish(start);
        }
    }

    /**
     * Transverses the children of a node.
     * 
     * @param node
     *            the parent node
     * @param visitor
     *            visitor for the children nodes
     */
    private void transverseChildren(AbstractNode node, Visitor visitor) {
        for (Edge edge : visitor.order(node, getEdges(node))) {
            AbstractNode adjacentNode = edge.getOpposite(node);

            if (visitor.discover(adjacentNode, edge)) {
                transverseChildren(adjacentNode, visitor);
                visitor.finish(adjacentNode);
            }
        }
    }
    
    private Collection<Edge> getEdges(AbstractNode node) {
        return ignoreDirection? node.getEdges() : node.getAdjacentEdges();
    }

}
