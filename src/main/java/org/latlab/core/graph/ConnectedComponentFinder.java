package org.latlab.core.graph;

import org.latlab.core.graph.search.DepthFirstSearch;
import org.latlab.core.graph.search.TimeVisitor;

import java.util.Set;

/**
 * For finding connected components in a graph.
 * @author leonard
 *
 */
public class ConnectedComponentFinder {
	/**
	 * Constructor.
	 * @param graph	graph in which connected components are to be found
	 */
	public ConnectedComponentFinder(UndirectedGraph graph) {
		search = new DepthFirstSearch(graph);
	}
	
	/**
	 * Returns the nodes connected to the specified node.
	 * @param node	the node connected to
	 * @return		nodes connected to the specified node
	 */
	public Set<AbstractNode> find(AbstractNode node) {
		TimeVisitor visitor = new TimeVisitor();
		search.perform(node, visitor);
		return visitor.discoveringTimes.keySet();
	}
	
	/**
	 * Returns the nodes connected to the specified node.
	 * @param node	the node connected to
	 * @param edge	an edge that should be ignored during search
	 * @return		nodes connected to the specified node
	 */
	public Set<AbstractNode> find(AbstractNode node, Edge ignoredEdge) {
		TimeVisitor visitor = new TimeVisitor();
		visitor.addIgnoredEdge(ignoredEdge);
		search.perform(node, visitor);
		return visitor.discoveringTimes.keySet();
	}
	
	private final DepthFirstSearch search;
}
