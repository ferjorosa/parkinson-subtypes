package org.latlab.core.reasoner;

import org.latlab.core.graph.AbstractNode;
import org.latlab.core.graph.Edge;
import org.latlab.core.graph.UndirectedGraph;
import org.latlab.core.learner.geast.Focus;
import org.latlab.core.model.*;
import org.latlab.core.util.ContinuousVariable;
import org.latlab.core.util.DiscreteVariable;
import org.latlab.core.util.SingularContinuousVariable;
import org.latlab.core.util.Variable;

import java.util.*;

/**
 * A natural clique tree. It is defined only for tree model currently. Also, it
 * does not support propagation on model with only one node.
 * 
 * @author leonard
 * 
 */
public class NaturalCliqueTree extends UndirectedGraph {
	/**
	 * Maps from variable to its corresponding separator.
	 */
	private Map<Variable, Separator> separators;

	/**
	 * Maps from variable to a clique that contains that variable. The key
	 * variable is child variable of a edge, except the root node. Different
	 * variables may refer to the same clique.
	 * 
	 * <p>
	 * Since a discrete variable may be mapped to a mixed clique (when there is
	 * only one discrete variable), the type {@code Clique} is used for discrete
	 * variable.
	 */
	private MixedVariableMap<Clique, MixedClique, Clique> cliques;

	/**
	 * Clique holding the root variable. It can be a {@code DiscreteClique} or
	 * {@code MixedClique}, depending on whether it has a discrete or continuous
	 * child variable.
	 */
	private Clique pivot;

	private void setPivot(Clique p) {
		if (pivot != null)
			pivot.setPivot(false);

		pivot = p;

		if (pivot != null)
			pivot.setPivot(true);
	}

	/**
	 * Constructs a natural clique tree from a tree model.
	 * 
	 * @param structure
	 *            model from which this clique tree is constructed from, only
	 *            the structure of the model (not parameters) is relevant
	 */
	public NaturalCliqueTree(TreeModel structure) {
		cliques =
				new MixedVariableMap<Clique, MixedClique, Clique>(
						structure.getNumberOfNodes());
		separators =
				new HashMap<Variable, Separator>(structure.getNumberOfNodes());

		// constructs the separators
		for (AbstractNode node : structure.getNodes()) {
			BeliefNode beliefNode = (BeliefNode) node;
			if (beliefNode.getDegree() > 1)
				addSeparator(beliefNode);
		}

		// constructs the cliques
		for (Edge edge : structure.getEdges()) {
			BeliefNode childNode = (BeliefNode) edge.getHead();
			BeliefNode parentNode = (BeliefNode) edge.getTail();

			Clique clique = addClique(parentNode, childNode);

			// connect the clique with each of the separators corresponding
			// to the edge vertices if those separators exist
			if (parentNode.getDegree() > 1) {
				addEdge(clique, separators.get(parentNode.getVariable()));
			}

			if (childNode.getDegree() > 1) {
				addEdge(separators.get(childNode.getVariable()), clique);
			}
		}

		// if the model contains only one node, add that single node as a clique
		if (structure.getNodes().size() == 1) {
			addClique((BeliefNode) structure.getNodes().get(0));
		}

		// use the clique that contains the root variable as the pivot
		DiscreteBeliefNode rootNode = (DiscreteBeliefNode) structure.getRoot();
		setPivot(cliques.get(rootNode.getVariable()));
		if (pivot() == null) {
			// if the root is not used as a key to any clique (when the tree
			// has more than one node), the clique of its first child is
			// used as the pivot, and the root variable is associated with
			// this pivot.
			BeliefNode childNode =
					(BeliefNode) rootNode.getChildren().iterator().next();
			setPivot(cliques.get(childNode.getVariable()));
			cliques.put(rootNode.getVariable(), pivot());
		}
	}

	/**
	 * Sets the focus of this propagation, so that the propagation stops at the
	 * boundaries of the subtree under focus.
	 * 
	 * @param focus
	 */
	public void setFocus(Focus focus) {
		assert focus != null;

		// set all the cliques to false first
		for (Clique clique : cliques.values()) {
			clique.setFocus(false);
		}

		// TODO LP: Should it check also whether any variable in a clique
		// is also under focus?

		// set the focus of a clique to true if its key variable it contains is
		// under focus
		for (Map.Entry<SingularContinuousVariable, MixedClique> entry : cliques.continuousMap().entrySet()) {
			if (focus.contains(entry.getKey()))
				entry.getValue().setFocus(true);
		}

		for (Map.Entry<DiscreteVariable, Clique> entry : cliques.discreteMap().entrySet()) {
			if (focus.contains(entry.getKey()))
				entry.getValue().setFocus(true);
		}

		// set the pivot to the clique holding the first variable if the focus
		// has more than one variables
		if (focus.size() > 0)
			setPivot(getClique(focus.pivot()));
	}

	/**
	 * Adds a separator node to this clique tree. Currently, only a discrete
	 * variable is possible for a separator node, since continuous variable must
	 * occur that the leaf nodes.
	 * 
	 * @param node
	 *            node from the original model that corresponds to the new
	 *            separator node
	 * @return the separator node created and added to this clique tree
	 */
	private Separator addSeparator(BeliefNode node) {
		DiscreteVariable variable = (DiscreteVariable) node.getVariable();
		Separator separator = new Separator(this, variable);
		addNode(separator);

		separators.put(variable, separator);
		return separator;
	}

	/**
	 * Adds a clique corresponding to the edge with the given {@code parent} and
	 * {@code child}. The {@code child} variable is used as the key variable of
	 * a clique.
	 * 
	 * @param parent
	 *            parent node of an edge
	 * @param child
	 *            child node of an edge
	 * @return clique created and added to this clique tree
	 */
	private Clique addClique(BeliefNode parent, BeliefNode child) {
		// the parent must be discrete
		final DiscreteBeliefNode discreteParent = (DiscreteBeliefNode) parent;

		final NaturalCliqueTree tree = this;

		// creates a different type of clique depending on the type of the child
		return child.accept(new BeliefNode.Visitor<Clique>() {
			public Clique visit(DiscreteBeliefNode node) {
				// add a discrete clique
				Clique clique =
						new DiscreteClique(tree, newCliqueName(),
								Arrays.asList(node.getVariable(),
										discreteParent.getVariable()));
				addNode(clique);
				cliques.put(node.getVariable(), clique);

				return clique;
			}

			public Clique visit(ContinuousBeliefNode node) {
				// add a mixed clique
				MixedClique mixedClique =
						new MixedClique(tree, newCliqueName(),
								node.getVariable(),
								discreteParent.getVariable());
				addNode(mixedClique);
				cliques.put(node.getVariable(), mixedClique);

				return mixedClique;
			}
		});
	}

	private Clique addClique(BeliefNode node) {
		// currently it doesn't network with only one node
		throw new UnsupportedOperationException();
	}

	private String newCliqueName() {
		return "Clique" + getNumberOfNodes();
	}

	/**
	 * Returns the separator corresponding to the given {@code variable}.
	 * 
	 * @param variable
	 *            of which the separator is to return
	 * @return separator corresponding to the given {@code variable}
	 */
	public Separator getSeparator(Variable variable) {
		return separators.get(variable);
	}

	/**
	 * Returns a clique containing the given {@code variable}, except the root
	 * variable.
	 * 
	 * <p>
	 * The root variable needs special handling because it may be contained only
	 * in a {@code MixedClique}.
	 * 
	 * @param variable
	 *            of which the clique is to return
	 * @return clique corresponding to the given {@code variable}
	 */
	public Clique getClique(DiscreteVariable variable) {
		return cliques.get(variable);
	}

	public MixedClique getClique(ContinuousVariable variable) {
		return cliques.get(variable);
	}

	public Clique getClique(Variable variable) {
		return cliques.get(variable);
	}

	/**
	 * Returns the pivot of this clique tree.
	 * 
	 * @return pivot of this clique tree
	 */
	public Clique pivot() {
		return pivot;
	}

	public Collection<Separator> separators() {
		return Collections.unmodifiableCollection(separators.values());
	}

	private class Subtree {
		// clique tree nodes in the current subtree and their corresponding
		// degrees
		final Map<CliqueTreeNode, Integer> subtree =
				new HashMap<CliqueTreeNode, Integer>(cliques.size());

		public Queue<Clique> initializeDegrees() {
			Queue<Clique> leaves = new LinkedList<Clique>();

			// initializes the degree map and put all leave nodes into the queue
			for (AbstractNode node : _nodes) {
				if (node instanceof CliqueTreeNode) {
					CliqueTreeNode treeNode = (CliqueTreeNode) node;

					int degree = treeNode.getDegree();
					subtree.put(treeNode, degree);

					// leaves must be clique nodes
					if (degree == 1)
						leaves.add((Clique) treeNode);
				}
			}

			return leaves;
		}

		public Separator getFirstNeighborSeparator(Clique clique) {
			// find the neighbor separator in the subtree,
			// there should be only one such neighbor
			for (AbstractNode node : clique.getNeighbors()) {
				if (subtree.containsKey(node)) {
					return (Separator) node;
				}
			}

			return null;
		}

		public int size() {
			return subtree.size();
		}

		/**
		 * Removes a node from the subtree and returns the original neighbors in
		 * the subtree.
		 * 
		 * @param node
		 * @return
		 */
		public Set<CliqueTreeNode> remove(CliqueTreeNode node) {
			Set<CliqueTreeNode> neighbors = new HashSet<CliqueTreeNode>();

			subtree.remove(node);

			for (AbstractNode neighbor : node.getNeighbors()) {
				Integer degree = subtree.get(neighbor);
				if (degree != null) {
					neighbors.add((CliqueTreeNode) neighbor);
					subtree.put((CliqueTreeNode) neighbor, degree - 1);
				}
			}

			return neighbors;
		}

		public int getDegree(CliqueTreeNode node) {
			Integer degree = subtree.get(node);
			return degree == null ? -1 : degree;
		}

		public Set<CliqueTreeNode> nodes() {
			return subtree.keySet();
		}
	}

	/**
	 * Find the minimal subtree that covers all the given variables.
	 * 
	 * @param variables
	 *            variables of interest
	 * @return a set of cliques contained in the subtree and a set of separators
	 *         in it
	 */
	public Set<CliqueTreeNode> findMinimalSubtree(
			Set<DiscreteVariable> variables) {
		// remove the irrelevant or redundant leaves while keeping the
		// remaining tree to cover all the variables of interest

		Subtree subtree = new Subtree();

		// leave nodes for examination in the current subtree tree
		Queue<Clique> leaves = subtree.initializeDegrees();

		// check nodes in the order of the queue
		while (!leaves.isEmpty()) {
			Clique leaf = leaves.remove();

			if (subtree.size() == 1) {
				if (Collections.disjoint(leaf.variables(), variables)) {
					subtree.remove(leaf);
				}

				break;
			}

			Separator neighborSeparator =
					subtree.getFirstNeighborSeparator(leaf);

			// check if this leaf contains any variable of interest but
			// not found in the subtree. If so, this clique is not redundant
			// and should be kept in the subtree.
			boolean redundant = true;
			for (Variable variable : leaf.variables()) {
				if (variable != neighborSeparator.variable()
						&& variables.contains(variable)) {
					redundant = false;
					break;
				}
			}

			// remove the leaf node from the subtree if it is redundant
			if (redundant) {
				Set<CliqueTreeNode> neighborSeparators = subtree.remove(leaf);

				for (CliqueTreeNode separator : neighborSeparators) {
					// if the neighbor separator is becomes a new leaf,
					// remove it
					if (subtree.getDegree(separator) == 1) {
						Set<CliqueTreeNode> neighborCliques =
								subtree.remove(separator);

						for (CliqueTreeNode clique : neighborCliques) {
							if (subtree.getDegree(clique) == 1) {
								leaves.add((Clique) clique);
							}
						}
					}
				}
			}
		}

		return subtree.nodes();
	}
}
