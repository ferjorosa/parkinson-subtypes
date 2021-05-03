/**
 * LoopyBeliefPropagation.java 
 * Copyright (C) 2007 Tao Chen, Kin Man Poon, Yi Wang, and Nevin L. Zhang
 */
package org.latlab.core.reasoner;

import org.latlab.core.graph.AbstractNode;
import org.latlab.core.graph.DirectedNode;
import org.latlab.core.model.BayesNet;
import org.latlab.core.model.DiscreteBeliefNode;
import org.latlab.core.util.DiscreteVariable;
import org.latlab.core.util.Function;
import org.latlab.core.util.Utils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;

/**
 * This class provides an implementation for Pearl's loopy belief propagation.
 * See Murphy et al. (1999) for a recent review and empirical study.
 * 
 * @author wangyi
 * 
 */
public class LoopyBeliefPropagation {

	/**
	 * The BN under query.
	 */
	private BayesNet _bayesNet;

	/**
	 * Maximum number of iterations. The default value is 100.
	 */
	private int _maxIter = 100;

	/**
	 * Convergence threshold. The default value is 1e-4.
	 */
	private double _tol = 1e-4;

	/**
	 * The array of exact beliefs. When this argument is specified, LBP will
	 * terminate once the average KL divergence between the exact beliefs and
	 * the approximate beliefs is lower than the desired value.
	 */
	private Function[] _exactBels;

	/**
	 * To be used along with _exactBels. If the average KL divergence between
	 * the exact beliefs and the approximate beliefs is smaller than this value,
	 * LBP terminates.
	 */
	private double _maxAverKl;

	/**
	 * Number of iterations.
	 */
	private int _iter;

	/**
	 * Messages for evidence.
	 */
	private HashMap<AbstractNode, Function> _evsMsgs;

	/**
	 * Old messages.
	 */
	private HashMap<AbstractNode, HashMap<AbstractNode, Function>> _oldMsgs;

	/**
	 * New messages.
	 */
	private HashMap<AbstractNode, HashMap<AbstractNode, Function>> _newMsgs;

	/**
	 * Old beliefs.
	 */
	private LinkedHashMap<AbstractNode, Function> _oldBels;

	/**
	 * New beliefs.
	 */
	private LinkedHashMap<AbstractNode, Function> _newBels;

	/**
	 * Constructs an LBP for the specified BN.
	 * 
	 * @param bayesNet
	 *            BN under query.
	 */
	public LoopyBeliefPropagation(BayesNet bayesNet) {
		_bayesNet = bayesNet;
		_evsMsgs = new HashMap<AbstractNode, Function>();
	}

	/**
	 * Returns <code>true</code> if any of the following three conditions
	 * satisfies: (1) # iterations exceeds the upper bound; (2) the accuracy of
	 * the approximate beliefs meets the requirement; (3) changes in beliefs are
	 * no greater than the threshold.
	 * 
	 * @return <code>true</code> if the stopping condition satisfies;
	 *         <code>false</code>, otherwise.
	 */
	private boolean checkStopCond() {
		// condition 1: max number of iterations
		if (_iter >= _maxIter) {
			return true;
		}

		// condition 2: accuracy requirement
		if (_exactBels != null && !_newBels.isEmpty()) {
			double kl = 0.0;
			AbstractNode node;

			for (Function bel : _exactBels) {
				node = _bayesNet.getNode(bel.getVariables().get(0));
				kl += Utils.computeKl(bel, _newBels.get(node));
			}

			if (kl / _exactBels.length <= _maxAverKl) {
				return true;
			}
		}

		// condition 3: difference in beliefs of two consecutive iterations
		// negative threshold disables this condition
		if (_tol < 0.0) {
			return false;
		}

		// at least run 2 iterations to compare beliefs
		if (_iter < 2) {
			return false;
		}

		for (AbstractNode node : _oldBels.keySet()) {
			Function oldBel = _oldBels.get(node);
			Function newBel = _newBels.get(node);
			double[] oldProb = oldBel.getCells();
			double[] newProb = newBel.getCells();
			for (int i = 0; i < oldProb.length; i++) {
				if (Math.abs(newProb[i] - oldProb[i]) > _tol) {
					return false;
				}
			}
		}

		return true;
	}

	/**
	 * Returns the belief for the specified query node.
	 * 
	 * @param query
	 *            The node whose belief is to be returned.
	 * @return The belief for the specified query node.
	 */
	public Function computeBelief(DiscreteBeliefNode query) {
		// return self-message for observed node
		if (_evsMsgs.containsKey(query)) {
			return _evsMsgs.get(query).clone();
		}

		return _newBels.get(query).clone();
	}

	/**
	 * Computes the product of all lambda messages sent to the specified node
	 * and the evidence message. Users are allowed to exclude one lambda message
	 * by setting the argument <code>exceptChild</code>.
	 * 
	 * @param node
	 *            The node for which the product is to be computed.
	 * @param nonZeroLambdaMsgProd
	 *            The product of all non-zero lambda messages and the evidence
	 *            message.
	 * @param excludedChildren
	 *            The list of children that send messages containing zeros.
	 * @param exceptChild
	 *            The child whose lambda message is to be excluded.
	 * @return The product.
	 */
	private Function computeAllLambdaMsgProd(DiscreteBeliefNode node,
			Function nonZeroLambdaMsgProd,
			ArrayList<AbstractNode> excludedChildren, AbstractNode exceptChild) {
		Function allLambdaMsgProd = nonZeroLambdaMsgProd.clone();
		for (AbstractNode child : excludedChildren) {
			if (child != exceptChild) {
				allLambdaMsgProd = allLambdaMsgProd.times(_oldMsgs.get(child)
						.get(node));
			}
		}
		return allLambdaMsgProd;
	}

	/**
	 * Computes the product of all pi messages sent to the specified node and
	 * the CPT of the node. Users are allowed to exclude one pi message by
	 * setting the argument <code>exceptParent</code>.
	 * 
	 * @param node
	 *            The node for which the product is to be computed.
	 * @param nonZeroPiMsgProd
	 *            The product of all non-zero pi messages.
	 * @param excludedParents
	 *            The list of parents that send messages containing zeros.
	 * @param exceptParent
	 *            The parent whose pi message is to be excluded.
	 * @return The product.
	 */
	private Function computeAllPiMsgProd(DiscreteBeliefNode node,
			Function nonZeroPiMsgProd, ArrayList<AbstractNode> excludedParents,
			AbstractNode exceptParent) {
		// no need to clone as it will be multiplied by CPT later in this method
		Function allPiMsgProd = nonZeroPiMsgProd;
		for (AbstractNode parent : excludedParents) {
			if (parent != exceptParent) {
				allPiMsgProd = allPiMsgProd.times(_oldMsgs.get(parent)
						.get(node));
			}
		}
		allPiMsgProd = allPiMsgProd.times(node.potential());
		return allPiMsgProd;
	}

	/**
	 * Computes the product of all lambda messages that do not contain zeros and
	 * the evidence messages if it exists.
	 * 
	 * @param node
	 *            The node for which the product is to be computed.
	 * @return The product.
	 */
	private Function computeNonZeroLambdaMsgProd(DiscreteBeliefNode node,
			ArrayList<AbstractNode> excludedChildren) {
		Function nonZeroLambdaMsgProd = Function.createIdentityFunction();

		// absorb evidence message
		if (_evsMsgs.containsKey(node)) {
			nonZeroLambdaMsgProd = nonZeroLambdaMsgProd.times(_evsMsgs
					.get(node));
		}

		// times up non-zero lambda messages
		for (DirectedNode child : node.getChildren()) {
			Function msg = _oldMsgs.get(child).get(node);
			if (msg.containsZeroCell()) {
				excludedChildren.add(child);
			} else {
				nonZeroLambdaMsgProd = nonZeroLambdaMsgProd.times(msg);
			}
		}

		return nonZeroLambdaMsgProd;
	}

	/**
	 * Computes the product of all pi messages that do not contain zeros.
	 * 
	 * @param node
	 *            The node for which the product is to be computed.
	 * @return The product.
	 */
	private Function computeNonZeroPiMsgProd(DiscreteBeliefNode node,
			ArrayList<AbstractNode> excludedParents) {
		Function nonZeroPiMsgProd = Function.createIdentityFunction();
		for (DirectedNode parent : node.getParents()) {
			Function msg = _oldMsgs.get(parent).get(node);
			if (msg.containsZeroCell()) {
				excludedParents.add(parent);
			} else {
				nonZeroPiMsgProd = nonZeroPiMsgProd.times(msg);
			}
		}
		return nonZeroPiMsgProd;
	}

	/**
	 * Returns the Bayes net.
	 * 
	 * @return The Bayes net.
	 */
	public BayesNet getBayesNet() {
		return _bayesNet;
	}

	/**
	 * Returns the maximum number of iterations.
	 * 
	 * @return The maximum number of iterations.
	 */
	public int getMaxIter() {
		return _maxIter;
	}

	/**
	 * Returns the stopping threshold for LBP.
	 * 
	 * @return The stopping threshold for LBP.
	 */
	public double getTol() {
		return _tol;
	}

	/**
	 * Initialize loopy belief propagation.
	 */
	private void initialize() {
		// reset # iterations
		_iter = 0;

		// initialize messages
		_oldMsgs = new HashMap<AbstractNode, HashMap<AbstractNode, Function>>();
		_newMsgs = new HashMap<AbstractNode, HashMap<AbstractNode, Function>>();
		for (AbstractNode node : _bayesNet.getNodes()) {
			DiscreteBeliefNode bNode = (DiscreteBeliefNode) node;
			DiscreteVariable var = bNode.getVariable();

			// old messages
			_oldMsgs.put(node, new HashMap<AbstractNode, Function>());

			// new messages
			HashMap<AbstractNode, Function> newMsgs = new HashMap<AbstractNode, Function>();
			_newMsgs.put(node, newMsgs);
			for (DirectedNode child : bNode.getChildren()) {
				newMsgs.put(child, Function.createUniformDistribution(var));
			}
			for (DirectedNode parent : bNode.getParents()) {
				newMsgs.put(parent, Function
						.createUniformDistribution(((DiscreteBeliefNode) parent)
								.getVariable()));
			}
		}

		// initialize beliefs
		_oldBels = new LinkedHashMap<AbstractNode, Function>();
		_newBels = new LinkedHashMap<AbstractNode, Function>();
	}

	/**
	 * Runs one iteration of LBP.
	 */
	private void parallelProtocol() {
		// increase # iterations
		_iter++;

		// update old messages and beliefs. note how we reuse the data
		// structure for old messages and beliefs to save time.
		HashMap<AbstractNode, HashMap<AbstractNode, Function>> tmpMsgs = _oldMsgs;
		_oldMsgs = _newMsgs;
		_newMsgs = tmpMsgs;
		LinkedHashMap<AbstractNode, Function> tmpBels = _oldBels;
		_oldBels = _newBels;
		_newBels = tmpBels;

		// data structures which contain children and parents that send messages
		// with zeros
		ArrayList<AbstractNode> excludedChildren = new ArrayList<AbstractNode>();
		ArrayList<AbstractNode> excludedParents = new ArrayList<AbstractNode>();

		// update messages for each node
		for (AbstractNode node : _bayesNet.getNodes()) {
			DiscreteBeliefNode bNode = (DiscreteBeliefNode) node;
			DiscreteVariable var = bNode.getVariable();

			// compute product of lambda messages from children
			excludedChildren.clear();
			Function nonZeroLambdaMsgProd = computeNonZeroLambdaMsgProd(bNode,
					excludedChildren);
			Function lambdaFunc = computeAllLambdaMsgProd(bNode,
					nonZeroLambdaMsgProd, excludedChildren, null);

			// compute product of pi messages from parents
			excludedParents.clear();
			Function nonZeroPiMsgProd = computeNonZeroPiMsgProd(bNode,
					excludedParents);
			Function allPiMsgProd = computeAllPiMsgProd(bNode,
					nonZeroPiMsgProd, excludedParents, null);
			Function piFunc = allPiMsgProd;
			for (DirectedNode parent : bNode.getParents()) {
				piFunc = piFunc.sumOut(((DiscreteBeliefNode) parent).getVariable());
			}

			// compute beliefs for hidden nodes
			if (!_evsMsgs.containsKey(node)) {
				Function bel = lambdaFunc.times(piFunc);
				bel.normalize();
				_newBels.put(node, bel);
			}

			// compute new pi messages to be sent to children
			HashMap<AbstractNode, Function> newMsgs = _newMsgs.get(node);
			for (DirectedNode child : bNode.getChildren()) {
				Function msg;
				if (excludedChildren.contains(child)) {
					msg = computeAllLambdaMsgProd(bNode, nonZeroLambdaMsgProd,
							excludedChildren, child);
				} else {
					msg = lambdaFunc.clone();
					msg.divide(_oldMsgs.get(child).get(node));
				}
				msg = msg.times(piFunc);
				msg.normalize();
				newMsgs.put(child, msg);
			}

			// compute new lambda messages to be sent to parents
			for (DirectedNode parent : bNode.getParents()) {
				Function msg;
				if (excludedParents.contains(parent)) {
					msg = computeAllPiMsgProd(bNode, nonZeroPiMsgProd,
							excludedParents, parent);
				} else {
					msg = allPiMsgProd.clone();
					msg.divide(_oldMsgs.get(parent).get(node));
				}
				for (DirectedNode parent2 : bNode.getParents()) {
					if (parent2 != parent) {
						msg = msg.sumOut(((DiscreteBeliefNode) parent2).getVariable());
					}
				}
				msg = msg.times(lambdaFunc);
				msg = msg.sumOut(var);
				msg.normalize();
				newMsgs.put(parent, msg);
			}
		}
	}

	/**
	 * Runs loopy belief propagation and returns the number of iterations.
	 * 
	 * @return the number of iterations.
	 */
	public int propagate() {
		// initialize
		initialize();

		// run parallel protocol until the stopping condition satisfies
		while (!checkStopCond()) {
			parallelProtocol();
		}

		return _iter;
	}

	/**
	 * Sets evidence.
	 * 
	 * @param vars
	 *            The observed variables.
	 * @param states
	 *            The observed states.
	 */
	public void setEvidence(DiscreteVariable[] vars, int[] states) {
		assert vars.length == states.length;

		// clear message for old evidence
		_evsMsgs.clear();

		// create messages for new evidence
		for (int i = 0; i < vars.length; i++) {
			DiscreteVariable var = vars[i];
			_evsMsgs.put(_bayesNet.getNode(var), Function
					.createIndicatorFunction(var, states[i]));
		}
	}

	/**
	 * Sets the array of exact beliefs.
	 * 
	 * @param exactBel
	 *            the array of exact beliefs.
	 */
	public void setExactBel(Function[] exactBel) {
		_exactBels = exactBel;
	}

	/**
	 * Sets the average KL divergence LBP has to achieve before its termination.
	 * 
	 * @param maxAverKl
	 *            the average KL divergence by which LBP terminates.
	 */
	public void setMaxAverKl(double maxAverKl) {
		_maxAverKl = maxAverKl;
	}

	/**
	 * Sets the maximum number of iterations.
	 * 
	 * @param maxIter
	 *            The new value of the maximum number of iterations.
	 */
	public void setMaxIter(int maxIter) {
		assert maxIter > 0;

		_maxIter = maxIter;
	}

	/**
	 * Sets the stopping threshold for LBP. Negative value disables this
	 * condition.
	 * 
	 * @param tol
	 *            The new value of the stopping threshold for LBP.
	 */
	public void setTol(double tol) {
		_tol = tol;
	}

}
