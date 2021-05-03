package org.latlab.core.graph.search;

import org.latlab.core.graph.AbstractGraph;
import org.latlab.core.graph.AbstractNode;
import org.latlab.core.graph.DirectedNode;
import org.latlab.core.graph.Edge;
import org.latlab.core.util.Algorithm;
import org.latlab.core.util.Caster;
import org.latlab.core.util.Pair;

import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

/**
 * Performs a breadth first search on a directed acyclic graph. A custom
 * operation based on the breadth first search can be implemented by extending
 * the {@code Visitor} class.
 * 
 * <p>
 * Note that the {@code finish} method is not called on the visitor.
 * 
 * @author leonard
 * 
 */
public class BreadthFirstSearch {
    private final AbstractGraph graph;

    /**
     * Queue of nodes to visit next.
     */
    private Queue<Pair<AbstractNode, Edge>> queue;

    /**
     * Constructor
     * 
     * @param graph
     *            a graph to search on
     */
    public BreadthFirstSearch(AbstractGraph graph) {
        this.graph = graph;
        this.queue = new LinkedList<Pair<AbstractNode, Edge>>();
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

        queue.clear();
        queue.add(new Pair<AbstractNode, Edge>(start, null));
        transverse(visitor);
    }

    /**
     * Performs the search. Repeated starts the search from all root nodes in
     * the graph.
     * 
     * @param visitor
     *            visitor for the nodes
     */
    public void perform(Visitor visitor) {
        queue.clear();

        List<DirectedNode> roots =
            Algorithm.filter(
                graph.getNodes(), new Caster<DirectedNode>(),
                DirectedNode.ROOT_PREDICATE);

        for (DirectedNode root : roots) {
            perform(root, visitor);
        }
    }

    /**
     * Transverses the nodes kept by the queue until the queue is empty. It adds
     * the children of a transversed node to the queue when it is first
     * discovered.
     * 
     * @param node
     *            the parent node
     * @param visitor
     *            visitor for the children nodes
     */
    private void transverse(Visitor visitor) {
        while (!queue.isEmpty()) {
            Pair<AbstractNode, Edge> pair = queue.remove();
            AbstractNode node = pair.first;
            if (visitor.discover(node, pair.second)) {
                for (Edge edge : visitor.order(node, node.getAdjacentEdges())) {
                    AbstractNode adjacentNode = edge.getOpposite(node);
                    queue.add(new Pair<AbstractNode, Edge>(adjacentNode, edge));
                }
            }
        }
    }

}
