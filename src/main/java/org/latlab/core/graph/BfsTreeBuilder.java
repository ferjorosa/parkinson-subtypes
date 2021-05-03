package org.latlab.core.graph;

import org.latlab.core.graph.search.AbstractVisitor;
import org.latlab.core.graph.search.BreadthFirstSearch;

public class BfsTreeBuilder {
	private final DirectedAcyclicGraph inputGraph;
	private final DirectedAcyclicGraph outputGraph;

	private BfsTreeBuilder(DirectedAcyclicGraph graph) {
		inputGraph = graph;
		outputGraph = new DirectedAcyclicGraph();
	}

	private void compute() {
		BreadthFirstSearch search = new BreadthFirstSearch(inputGraph);
		search.perform(new TreeBuildingVisitor());
	}

	public static DirectedAcyclicGraph compute(DirectedAcyclicGraph graph) {
		BfsTreeBuilder builder = new BfsTreeBuilder(graph);
		builder.compute();
		return builder.outputGraph;
	}

	/**
	 * Builds a tree based on the nodes and edges visited from a search.
	 * 
	 * @author leonard
	 * 
	 */
	private class TreeBuildingVisitor extends AbstractVisitor {

		/**
		 * Adds the node in the output graph and an edge from
		 */
		public boolean discover(AbstractNode node, Edge edge) {
			if (outputGraph.containsNode(node.getName()))
				return false;

			AbstractNode newNode = outputGraph.addNode(node.getName());

			if (edge != null) {
				AbstractNode adjacentNode = outputGraph.getNode(edge
						.getOpposite(node).getName());

				outputGraph.addEdge(newNode, adjacentNode);
			}

			return true;
		}

		public void finish(AbstractNode node) {
		}

	}
}
