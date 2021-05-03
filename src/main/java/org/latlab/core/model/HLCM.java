/**
 * HLCM.java 
 * Copyright (C) 2006 Tao Chen, Kin Man Poon, Yi Wang, and Nevin L. Zhang
 */
package org.latlab.core.model;

import org.latlab.core.graph.AbstractNode;
import org.latlab.core.graph.DirectedNode;
import org.latlab.core.graph.Edge;
import org.latlab.core.util.DataSet;
import org.latlab.core.util.DiscreteVariable;
import org.latlab.core.util.Function;

import java.util.*;

/**
 * This class provides an implementation for HLC models.
 * 
 * @author Tao Chen
 * 
 */
public class HLCM extends TreeModel {

	/**
	 * the prefix of default names of HLCMs.
	 */
	private final static String NAME_PREFIX = "HLCM";

	/**
	 * the number of created HLCMs.
	 */
	private static int _count = 0;

	/**
	 * Store the unitIncrease value.
	 */
	private double _unitIncrease;

	/**
	 * Constructs an empty HLCM with the specified name.
	 * 
	 * @param name
	 *            name of this BN.
	 */
	public HLCM(String name) {
		super(name);
		_count++;
	}

	/**
	 * Default constructor.
	 */
	public HLCM() {
		this(NAME_PREFIX + _count);
	}

	/**
	 * TODO This method override BayesNet.clone(), however, they are quite
	 * similar. Creates and returns a deep copy of this HLCM. This
	 * implementation copies everything in this HLCM but the name and variables.
	 * The default name will be used for the copy instead of the original one.
	 * The variables will be reused other than deeply copied. This will
	 * facilitate learning process. However, one cannot change node names after
	 * clone. TODO avoid redundant operations on CPTs.
	 * <p>
	 * Also note that cpts are also cloned.
	 * </p>
	 * 
	 * @return a deep copy of this HLCM.
	 */
	@SuppressWarnings("unchecked")
	public HLCM clone() {
		HLCM copy = new HLCM();

		// copies nodes
		for (AbstractNode node : _nodes) {
			copy.addNode(((DiscreteBeliefNode) node).getVariable());
		}

		// copies edges
		for (Edge edge : _edges) {
			copy.addEdge(copy.getNode(edge.getHead().getName()), copy
					.getNode(edge.getTail().getName()));
		}
		// copies CPTs
		for (AbstractNode node : copy._nodes) {
			DiscreteBeliefNode bNode = (DiscreteBeliefNode) node;
			bNode.setCpt(getNode(bNode.getVariable()).potential().clone());
		}

		// copies loglikelihoods
		copy._loglikelihoods = (HashMap<DataSet, Double>) _loglikelihoods
				.clone();

		return copy;
	}

	/**
	 * Create an LCM from maniVars. The parameters are randomly set.
	 * 
	 * @param maniVars
	 *            manifest variables
	 * @param cardinality
	 *            The cardinality of the latent variable
	 * @return The resulting LCM.
	 */
	public static HLCM createLCM(DiscreteVariable[] maniVars, int cardinality) {

		HLCM lCM = new HLCM();
		DiscreteBeliefNode root = lCM.addNode(new DiscreteVariable(cardinality));
		for (DiscreteVariable var : maniVars) {
			lCM.addEdge(lCM.addNode(var), root);
		}
		lCM.randomlyParameterize();

		return lCM;
	}

	/**
	 * When use this method, we suppose that there is a non-leave(latent)
	 * BeliefeNode which is of Variable var.
	 * <p>
	 * An equivalent HLCM is returned which is rooted at that node. The original
	 * HLCM is immutable and the output HLCM is a clone.
	 * </p>
	 * 
	 * @param var
	 * @return An HLCM after root walking.
	 */
	public HLCM changeRoot(DiscreteVariable var) {

		HLCM equiModel = (HLCM) clone();
		DiscreteBeliefNode newRoot = equiModel.getNode(var);

		assert newRoot != null && !newRoot.isLeaf();

		// collect all (copies) of latent nodes from
		// newRootInModel to root (excluding root) into Stack
		Stack<DiscreteBeliefNode> path = path2Root(newRoot);
		path.pop();

		while (!path.empty()) {
			DiscreteBeliefNode nextRoot = path.pop();
			equiModel.shiftRoot(nextRoot);
		}

		return equiModel;
	}

	/**
	 * 
	 * @param latVar
	 *            A Variable in the model whose states will be merged.
	 * @param newVar
	 *            The new Variable whose cardinality is decreased by one and
	 *            will replace latVar
	 * @return Possible models resulted by state merge
	 */
	/**
	 * @param latVar
	 * @param newVar
	 * @return
	 */
	public HLCM[] modelsByStateMerge(DiscreteVariable latVar, DiscreteVariable newVar) {

		DiscreteBeliefNode latNode = getNode(latVar);
		HashMap<DiscreteVariable, Function> originalChildren = new HashMap<DiscreteVariable, Function>();

		Function originalParent = latNode.potential();
		for (DirectedNode node : latNode.getChildren()) {
			originalChildren.put(((DiscreteBeliefNode) node).getVariable(),
					((DiscreteBeliefNode) node).potential());
		}

		HLCM template = modelByVarReplacement(latVar, newVar);

		int cardinality = latVar.getCardinality();
		HLCM[] modelList = new HLCM[(cardinality * (cardinality - 1)) / 2];
		int index = 0;
		for (int i = 0; i < cardinality - 1; i++)
			for (int j = i + 1; j < cardinality; j++) {

				HLCM model = template.clone();
				DiscreteBeliefNode newNode = model.getNode(newVar);
				newNode.setCpt(originalParent.combine(latVar, i, j, newVar));

				for (DirectedNode node : newNode.getChildren()) {
					DiscreteBeliefNode bNode = (DiscreteBeliefNode) node;
					bNode.setCpt(originalChildren.get(bNode.getVariable())
							.averageCombine(latVar, i, j, newVar));
				}

				modelList[index] = model;
				index++;
			}

		return modelList;
	}

	/**
	 * 
	 * @param latVar
	 *            A Variable in the model whose states will be Splited.
	 * @param newVar
	 *            The new Variable whose cardinality is increased by one and
	 *            will replace latVar
	 * @return Possible models resulted by state split
	 */
	public HLCM[] modelsByStateSplit(DiscreteVariable latVar, DiscreteVariable newVar) {

		DiscreteBeliefNode latNode = getNode(latVar);
		HashMap<DiscreteVariable, Function> originalChildren = new HashMap<DiscreteVariable, Function>();

		Function originalParent = latNode.potential();
		for (DirectedNode node : latNode.getChildren()) {
			originalChildren.put(((DiscreteBeliefNode) node).getVariable(),
					((DiscreteBeliefNode) node).potential());
		}

		HLCM template = modelByVarReplacement(latVar, newVar);

		int cardinality = latVar.getCardinality();
		HLCM[] modelList = new HLCM[cardinality];
		for (int i = 0; i < cardinality; i++) {
			HLCM model = template.clone();
			DiscreteBeliefNode newNode = model.getNode(newVar);
			newNode.setCpt(originalParent.split(latVar, i, newVar));

			for (DirectedNode node : newNode.getChildren()) {
				DiscreteBeliefNode bNode = (DiscreteBeliefNode) node;
				bNode.setCpt(originalChildren.get(bNode.getVariable())
						.stateCopy(latVar, i, newVar));
			}
			modelList[i] = model;
		}

		return modelList;
	}

	/**
	 * Replace the latVar by the newVar. Return the new model while this model
	 * has no change. The parameter of the new node and children of the new node
	 * are not refined.
	 * 
	 * @param latVar
	 * @param newVar
	 * @return
	 */
	public HLCM modelByVarReplacement(DiscreteVariable latVar, DiscreteVariable newVar) {

		HLCM template = (HLCM) clone();
		DiscreteBeliefNode latNode = template.getNode(latVar);
		DiscreteBeliefNode newNode = template.addNode(newVar);

		Set<DirectedNode> children = latNode.getChildren();
		DiscreteBeliefNode parent = (DiscreteBeliefNode) latNode.getParent();

		for (DirectedNode child : children) {
			DiscreteBeliefNode beliefChild = (DiscreteBeliefNode) child;
			template.addEdge(beliefChild, newNode);
		}

		if (parent != null) {
			template.addEdge(newNode, parent);
		}

		template.removeNode(latNode);

		return template;
	}

	/**
	 * v1 and v2 are two neighbors of latVar. We introduce a newNode with
	 * Variable newVar for v1 and v2. newVar has the same cardinality as latVar.
	 * The parameters are set properly depended on the relationship between
	 * latVar and v1, v2.
	 * 
	 * @param latVar
	 * @param v1
	 * @param v2
	 * @param newVar
	 * @return
	 */
	public HLCM introduceNewNode(DiscreteVariable latVar, DiscreteVariable v1, DiscreteVariable v2,
			DiscreteVariable newVar) {
		assert getNode(latVar) != null && getNode(v1) != null
				&& getNode(v2) != null;

		HLCM template = (HLCM) clone();
		// We operate in template.

		DiscreteBeliefNode latNode = template.getNode(latVar);
		DiscreteBeliefNode neighbor1 = template.getNode(v1);
		DiscreteBeliefNode neighbor2 = template.getNode(v2);
		DiscreteBeliefNode newNode = template.addNode(newVar);
		if (neighbor1.getParent() == latNode
				&& neighbor2.getParent() == latNode) {
			// Function cpt1 = neighbor1.getCpt();
			// Function cpt2 = neighbor2.getCpt();

			template.removeEdge(latNode.getEdge(neighbor1));
			template.removeEdge(latNode.getEdge(neighbor2));
			template.addEdge(newNode, latNode);
			template.addEdge(neighbor1, newNode);
			template.addEdge(neighbor2, newNode);

			// neighbor1.setCpt(cpt1.replaceVar(latVar, newVar));
			// neighbor2.setCpt(cpt2.replaceVar(latVar, newVar));
			// newNode
			// .setCpt(Function
			// .createDeterCondDistribution(latVar, newVar));
		} else {
			DiscreteBeliefNode parNeighbor = latNode.getParent() == neighbor1 ? neighbor1
					: neighbor2;
			DiscreteBeliefNode chiNeighbor = latNode.getParent() == neighbor1 ? neighbor2
					: neighbor1;
			// Function cpt1 = latNode.getCpt();
			// Function cpt2 = chiNeighbor.getCpt();

			template.removeEdge(latNode.getEdge(parNeighbor));
			template.removeEdge(latNode.getEdge(chiNeighbor));
			template.addEdge(newNode, parNeighbor);
			template.addEdge(latNode, newNode);
			template.addEdge(chiNeighbor, newNode);

			// newNode.setCpt(cpt1.replaceVar(latVar, newVar));
			// chiNeighbor.setCpt(cpt2.replaceVar(latVar, newVar));
			// latNode
			// .setCpt(Function
			// .createDeterCondDistribution(latVar, newVar));
		}

		return template;

	}

	/**
	 * Suppose that child node is a latent child of the root in the current
	 * HLCM. Current HLCM never change. We clone a new HLCM and there is such a
	 * corresponding child node. Then we delete it from the new HLCM and make
	 * children of this node children of the root.
	 * <p>
	 * Parameters are inherited except P(new child | root) for every new child.
	 * </p>
	 * 
	 * @param child
	 * @return A cloned HLCM in which the corresponding child node has been
	 *         deleted.
	 */
	public HLCM deleteLatChildofRoot(DiscreteBeliefNode child) {

		assert this.containsNode(child);
		assert child.getParent().isRoot();
		assert !child.isLeaf();

		HLCM candModel = (HLCM) this.clone();
		// Operations take place in the cloned candModel
		child = candModel.getNode(child.getVariable());
		DiscreteBeliefNode root = (DiscreteBeliefNode) child.getParent();
		Set<DirectedNode> grandChildren = child.getChildren();

		candModel.removeNode(child);
		for (DirectedNode grandChild : grandChildren) {
			candModel.addEdge(grandChild, root);
			((DiscreteBeliefNode) grandChild).randomlyParameterize();
		}

		return candModel;
	}

	/**
	 * Return the set of hidden Variables. Note that the reference to these
	 * variable are contained in the Set.
	 * 
	 * @return The set of hidden Variables.
	 */
	public Set<DiscreteVariable> getLatVars() {
		Set<DiscreteVariable> latVars = new HashSet<DiscreteVariable>();
		for (DiscreteVariable var : getDiscreteVariables()) {
			if (!getNode(var).isLeaf())
				latVars.add(var);
		}
		return latVars;
	}

	/**
	 * Return the set of manifest Variables. Note that the reference to these
	 * variable are contained in the Set.
	 * 
	 * @return The set of manifest(leaf) Variables.
	 */
	public Set<DiscreteVariable> getManifestVars() {
		Set<DiscreteVariable> manifestVars = new HashSet<DiscreteVariable>();
		for (DiscreteVariable var : getDiscreteVariables()) {
			if (getNode(var).isLeaf())
				manifestVars.add(var);
		}
		return manifestVars;
	}

	/**
	 * Whether the model satisfies the regularity of cardinality.
	 * 
	 * @return
	 */
	public boolean isCardRegular() {
		boolean regular = true;
		for (AbstractNode node : getNodes()) {
			DiscreteBeliefNode bNode = (DiscreteBeliefNode) node;
			if (bNode.getVariable().getCardinality() > bNode
					.computeMaxPossibleCardInHLCM()) {
				regular = false;
				break;
			}
		}
		return regular;
	}

	public boolean isCardRegular(DiscreteVariable var) {

		assert getNode(var) != null;
		DiscreteBeliefNode node = getNode(var);
		if (node.isLeaf())
			return true;

		int maxPossibleCard = node.computeMaxPossibleCardInHLCM();
		int cardinality = node.getVariable().getCardinality();
		if (cardinality > maxPossibleCard)
			return false;
		else
			return true;
	}

	/**
	 * Note that the collection of vars may contain manifest variables. For
	 * manifest variable, no need to check the regularity.
	 * 
	 * @param vars
	 * @return
	 */
	public boolean isCardRegular(Collection<DiscreteVariable> vars) {
		boolean regular = true;
		for (DiscreteVariable var : vars) {
			if (!isCardRegular(var)) {
				regular = false;
				break;
			}
		}
		return regular;
	}

	/**
	 * Get root node of this HLCM. Return null for an empty HLCM.
	 * 
	 * @return Root node.
	 */
	public DiscreteBeliefNode getRoot() {

		LinkedList<AbstractNode> allNodes = getNodes();
		// return null for an empty HLCM.
		if (allNodes.isEmpty())
			return null;

		DirectedNode oneNode = (DirectedNode) allNodes.getFirst();
		while (true) {
			if (oneNode.isRoot())
				break;
			oneNode = oneNode.getParent();
		}
		return (DiscreteBeliefNode) oneNode;
	}

	/**
	 * 
	 * @return unitIncrease
	 */
	public double getUnitIncrease() {
		return _unitIncrease;
	}

	/**
	 * Set unitIncrease.
	 */
	public void setUnitIncrease(double unitIncrease) {
		_unitIncrease = unitIncrease;
	}

	/**
	 * Suppose that child1 and child2 are children of the root node. Return a
	 * clone of this model. In the cloned model, there are corresponding child1
	 * and child2. All operations take place in the cloned model: A latent node
	 * is introduced as the new parent of child1 and child2. This new latent
	 * node is a new child of the root. The new latent node is of cardinality 2.
	 * <p>
	 * Parameters are inherited except P(child1|newly introduced Node ),
	 * P(child2|newly introduced Node ), P(newly introduced Node | root),
	 * </p>
	 * 
	 * @param child1
	 *            A child node of the root
	 * @param child2
	 *            A child node of the root
	 * @return A cloned model in which a new latent node is introduced as parent
	 *         of child1 and child2.
	 */
	public HLCM introduceParent4Siblings(DiscreteBeliefNode child1, DiscreteBeliefNode child2) {

		assert this.containsNode(child1);
		assert this.containsNode(child2);
		assert child1.getParent().isRoot();
		assert child2.getParent().isRoot();

		HLCM candModel = (HLCM) this.clone();
		child1 = candModel.getNode(child1.getVariable());
		child2 = candModel.getNode(child2.getVariable());
		DiscreteBeliefNode root = candModel.getRoot();

		candModel.removeEdge(child1.getEdge(root));
		candModel.removeEdge(child2.getEdge(root));
		DiscreteBeliefNode newNode = candModel.addNode(new DiscreteVariable(2));

		candModel.addEdge(child1, newNode);
		child1.randomlyParameterize();

		candModel.addEdge(child2, newNode);
		child2.randomlyParameterize();

		candModel.addEdge(newNode, root);
		newNode.randomlyParameterize();

		return candModel;
	}

	/**
	 * Return a clone of this model. The calling model has no change. For the
	 * returned model, all thing are the same except that the root node is
	 * replaced by another one which has one more state than the original root.
	 * <p>
	 * Paramters are inherited except P(newRoot), P(Child|newRoot) for every
	 * child. These parameters are assigned but We suppose they are
	 * insignificant, that is, never been use in the future.
	 * </p>
	 * 
	 * @return A cloned HLCM in which the root canotains one more state
	 */
	public HLCM introduceState4Root() {

		HLCM candModel = (HLCM) this.clone();

		DiscreteBeliefNode rootNode = candModel.getRoot();
		DiscreteVariable rootVar = rootNode.getVariable();
		Set<DirectedNode> children = rootNode.getChildren();

		DiscreteBeliefNode newRootNode = candModel.addNode(new DiscreteVariable(rootVar
				.getCardinality() + 1));
		candModel.removeNode(rootNode);

		for (DirectedNode child : children) {
			DiscreteBeliefNode beliefChild = (DiscreteBeliefNode) child;

			candModel.addEdge(beliefChild, newRootNode);
			// Be careful. ctp need work.
			// But now I find no need
			// beliefChild.randomlyParameterize();
		}
		// Be careful. ctp need work.
		// But now I find no need
		// newRootNode.randomlyParameterize();

		return candModel;

	}

	/**
	 * Return the nodes along the path from var1 to var2 in the order.
	 * 
	 * @param var1
	 * @param var2
	 * @return
	 */
	public ArrayList<DiscreteBeliefNode> computePath(DiscreteVariable var1, DiscreteVariable var2) {

		assert getNode(var1) != null && getNode(var2) != null;

		Stack<DiscreteBeliefNode> NodeOne2Root = path2Root(getNode(var1));
		Stack<DiscreteBeliefNode> NodeTwo2Root = path2Root(getNode(var2));

		DiscreteBeliefNode sameNode = null;
		while (!NodeOne2Root.empty() && !NodeTwo2Root.empty()) {
			DiscreteBeliefNode node1 = NodeOne2Root.pop();
			DiscreteBeliefNode node2 = NodeTwo2Root.pop();
			if (node1 == node2) {
				sameNode = node1;
			} else {
				NodeOne2Root.push(node1);
				NodeTwo2Root.push(node2);
				break;
			}
		}

		ArrayList<DiscreteBeliefNode> path = new ArrayList<DiscreteBeliefNode>();
		while (!NodeOne2Root.empty()) {
			path.add(NodeOne2Root.pop());
		}

		Collections.reverse(path);

		path.add(sameNode);

		while (!NodeTwo2Root.empty()) {
			path.add(NodeTwo2Root.pop());
		}

		return path;
	}

	/**
	 * Compute the path from this node to Root. All BeliefNode along this path
	 * will be returned. The root is also included.
	 * 
	 * @param node
	 * @return All BeliefNodes from the node to Root.
	 */
	public Stack<DiscreteBeliefNode> path2Root(DiscreteBeliefNode node) {

		assert this.containsNode(node);

		Stack<DiscreteBeliefNode> path = new Stack<DiscreteBeliefNode>();
		path.add(node);
		while (!node.isRoot()) {
			node = (DiscreteBeliefNode) node.getParent();
			path.push(node);
		}
		return path;
	}

	/**
	 * When use this method, make sure that childOfRoot is one BeliefNode in
	 * this model. Moreover, it must be a latent child of the root.
	 * <p>
	 * The method shift root to childOfRoot, i.e. only shift one step.
	 * </p>
	 * 
	 * @param childOfRoot
	 *            A latent child of the root
	 */
	private void shiftRoot(DiscreteBeliefNode childOfRoot) {

		DiscreteBeliefNode curRoot = (DiscreteBeliefNode) childOfRoot.getParent();

		Function joint = curRoot.potential().times(childOfRoot.potential());

		Function newPr = joint.marginalize(childOfRoot.getVariable());

		// test to see if P(c)=ZERO for some state of c
		// Nevin's hlcm has this test. There ZERO is a very small value.
		// if (newPr.min() < ZERO)
		// return false;

		joint.normalize(curRoot.getVariable());
		Function newPc = joint;

		// Note: One must first change structure
		removeEdge(childOfRoot.getEdge(curRoot));
		addEdge(curRoot, childOfRoot);
		// Then deal with cpts
		curRoot.setCpt(newPc);
		childOfRoot.setCpt(newPr);

	}
}