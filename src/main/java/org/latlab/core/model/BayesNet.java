/**
 * BayesNet.java Copyright (C) 2006 Tao Chen, Kin Man Poon, Yi Wang, and Nevin
 * L. Zhang
 */
package org.latlab.core.model;

import org.latlab.core.graph.AbstractNode;
import org.latlab.core.graph.DirectedAcyclicGraph;
import org.latlab.core.graph.DirectedNode;
import org.latlab.core.graph.Edge;
import org.latlab.core.util.*;

import java.io.*;
import java.util.*;

/**
 * This class provides an implementation for Bayes nets (BNs).
 * 
 * @author Yi Wang
 * 
 */
public class BayesNet extends DirectedAcyclicGraph {

	/**
	 * the prefix of default names of BNs.
	 */
	private final static String NAME_PREFIX = "BayesNet";

	/**
	 * the number of created BNs.
	 */
	private static int _count = 0;

	/**
	 * Creates a BN that is defined by the specified file.
	 * 
	 * @param file
	 *            file that defines a BN.
	 * @return the BN defined by the specified file.
	 * @throws IOException
	 *             if an I/O error occurs.
	 */
	public final static BayesNet createBayesNet(String file) throws IOException {
		BayesNet bayesNet = new BayesNet(createDefaultName());

		// only supports BIF format by now
		bayesNet.loadBif(file);

		return bayesNet;
	}

	/**
	 * Returns the default name for the next BN.
	 * 
	 * @return the default name for the next BN.
	 */
	public final static String createDefaultName() {
		return NAME_PREFIX + _count;
	}

	/**
	 * the name of this BN.
	 */
	protected String _name;

	protected MixedVariableMap<BeliefNode, ContinuousBeliefNode, DiscreteBeliefNode> _variables;

	/**
	 * the map from data sets to loglikelihoods of this BN on them.
	 * loglikelihoods will expire once the structure or the parameters of this
	 * BN change.
	 */
	protected HashMap<DataSet, Double> _loglikelihoods;

	/**
	 * Constructs an empty BN.
	 * 
	 */
	public BayesNet() {
		this(createDefaultName());
	}

	/**
	 * Constructs an empty BN with the specified name.
	 * 
	 * @param name
	 *            name of this BN.
	 */
	public BayesNet(String name) {
		super();

		name = name.trim();

		// name cannot be blank
		assert name.length() > 0;

		_name = name;
		_variables =
				new MixedVariableMap<BeliefNode, ContinuousBeliefNode, DiscreteBeliefNode>();
		_loglikelihoods = new HashMap<DataSet, Double>();

		_count++;
	}

	protected BayesNet(BayesNet other) {
		this();

		// copies nodes
		for (AbstractNode node : other._nodes) {
			addNode(((BeliefNode) node).getVariable());
		}

		// copies edges
		for (Edge edge : other._edges) {
			try {
				addEdge(getNode(edge.getHead().getName()),
						getNode(edge.getTail().getName()));
			} catch (RuntimeException e) {
				throw e;
			}
		}

		// copies CPTs
		for (AbstractNode node : _nodes) {
			BeliefNode beliefNode = (BeliefNode) node;
			BeliefNode otherNode = other.getNode(beliefNode.getVariable());
			beliefNode.setPotential(otherNode.potential().clone());
		}

		// copies loglikelihoods
		_loglikelihoods = new HashMap<DataSet, Double>(other._loglikelihoods);
	}

	/**
	 * Adds an edge that connects the two specified nodes to this BN and returns
	 * the edge. This implementation extends
	 * <code>AbstractGraph.addEdge(AbstractNode, AbstractNode)</code> such that
	 * all loglikelihoods will be expired.
	 * 
	 * The resulting edge is {@code head <- tail}.
	 * 
	 * @param head
	 *            head of the edge.
	 * @param tail
	 *            tail of the edge.
	 * @return the edge that was added to this BN.
	 */
	public final Edge addEdge(AbstractNode head, AbstractNode tail) {
		Edge edge = super.addEdge(head, tail);

		// loglikelihoods expire
		expireLoglikelihoods();

		return edge;
	}

	/**
	 * This implementation is no long supported in <code>BayesNet</code>. Use
	 * <code>addNode(Variable)</code> instead.
	 * 
	 * @see addNode(Variable)
	 */
	public final DiscreteBeliefNode addNode(String name)
			throws UnsupportedOperationException {
		throw new UnsupportedOperationException();
	}

	public BeliefNode addNode(Variable variable) {
		return variable.accept(new Variable.Visitor<BeliefNode>() {

			@Override
			public BeliefNode visit(DiscreteVariable variable) {
				return addNode(variable);
			}

			@Override
			public BeliefNode visit(JointContinuousVariable variable) {
				return addNode(variable);
			}

			@Override
			public BeliefNode visit(SingularContinuousVariable variable) {
				return addNode(variable);
			}

		});
	}

	/**
	 * Adds a node with the specified variable attached to this BN and returns
	 * the node.
	 * 
	 * @param variable
	 *            variable to be attached to the node.
	 * @return the node that was added to this BN.
	 */
	public final DiscreteBeliefNode addNode(DiscreteVariable variable) {
		DiscreteBeliefNode node = new DiscreteBeliefNode(this, variable);
		addNode(node);
		_variables.put(variable, node);
		return node;
	}

	public ContinuousBeliefNode addNode(SingularContinuousVariable variable) {
		return addContinuousNode(new ContinuousBeliefNode(this, variable));
	}

	public ContinuousBeliefNode addNode(JointContinuousVariable variable) {
		return addContinuousNode(new ContinuousBeliefNode(this, variable));
	}

	private ContinuousBeliefNode addContinuousNode(ContinuousBeliefNode node) {
		addNode(node);
		_variables.put(node.getVariable(), node);
		return node;
	}

	public ContinuousBeliefNode combine(boolean connectNewNode,
			ContinuousBeliefNode node1, ContinuousBeliefNode node2) {
		return combine(connectNewNode, Arrays.asList(node1, node2));
	}

	/**
	 * Combines the given collection of nodes into a new node. It removes those
	 * given nodes from the network and adds the merge node into the network.
	 * <p>
	 * It connects the new node to the original neighbors
	 * 
	 * @param connectNewNode
	 * @param nodes
	 * @return
	 */
	public ContinuousBeliefNode combine(boolean connectNewNode,
			List<ContinuousBeliefNode> nodes) {
		if (nodes.size() < 2)
			return nodes.get(0);

		Set<DirectedNode> parents = Collections.emptySet();
		Set<DirectedNode> children = Collections.emptySet();

		TreeSet<SingularContinuousVariable> variables =
				new TreeSet<SingularContinuousVariable>();

		// it removes all the other nodes first so that they will not appear in
		// the parents and children list
		for (int i = 1; i < nodes.size(); i++) {
			ContinuousBeliefNode node = nodes.get(i);
			variables.addAll(node.getVariable().variables());
			removeNode(node);
		}

		// need to make copies because removing the base node will change these
		if (connectNewNode) {
			parents = new HashSet<DirectedNode>(nodes.get(0).getParents());
			children = new HashSet<DirectedNode>(nodes.get(0).getChildren());
		}

		variables.addAll(nodes.get(0).getVariable().variables());
		removeNode(nodes.get(0));

		ContinuousBeliefNode newNode =
				addNode(JointContinuousVariable.attach(variables));

		if (connectNewNode) {
			for (DirectedNode child : children) {
				addEdge(child, newNode);
			}

			for (DirectedNode parent : parents) {
				addEdge(newNode, parent);
			}
		}

		// the likelihoods are expired by the previous functions so it needs
		// not be called again.

		return newNode;
	}

	//
	// /**
	// * Creates a new node by combining the {@code base} node to the {@code
	// * other} node. The new node will have the same neighbors as the {@code
	// * base} node.
	// *
	// * @param base
	// * @param other
	// * @return the combined new node
	// */
	// public ContinuousBeliefNode combine(
	// ContinuousBeliefNode base, ContinuousBeliefNode other) {
	// // need to make copies because removing the base node will change these
	// Set<DirectedNode> children =
	// new HashSet<DirectedNode>(base.getChildren());
	// Set<DirectedNode> parents =
	// new HashSet<DirectedNode>(base.getParents());
	//
	// removeNode(base);
	// removeNode(other);
	//
	// ContinuousBeliefNode newNode =
	// addNode(new JointContinuousVariable(base.getVariable(), other
	// .getVariable()));
	//
	// for (DirectedNode child : children) {
	// addEdge(child, newNode);
	// }
	//
	// for (DirectedNode parent : parents) {
	// addEdge(newNode, parent);
	// }
	//
	// // the likelihoods are expired by the previous functions so it needs
	// // not be called again.
	//
	// _variables.put(newNode.getVariable(), newNode);
	// return newNode;
	// }

	/**
	 * Separates the given variables from the base node. The base node is
	 * deleted, and two new nodes with the partitioned variables are added.
	 * Returns the the separated nodes in a pair, in which the first node holds
	 * the reduced set of variables, and the second node the separated set of
	 * variables.
	 * <p>
	 * It connects the two new nodes to the same neighbors of the old node if
	 * requested.
	 * 
	 * @param connectNewNodes
	 * @param base
	 * @param variables
	 * @return
	 */
	public Pair<ContinuousBeliefNode, ContinuousBeliefNode> separate(
			boolean connectNewNodes, JointContinuousVariable base,
			Collection<SingularContinuousVariable> variables) {

		Set<DirectedNode> parents = Collections.emptySet();
		Set<DirectedNode> children = Collections.emptySet();

		ContinuousBeliefNode baseNode = getNode(base);
		if (connectNewNodes) {
			parents = new HashSet<DirectedNode>(baseNode.getParents());
			children = new HashSet<DirectedNode>(baseNode.getChildren());
		}

		// form the reduced set of variables
		Collection<SingularContinuousVariable> baseVariables =
				new HashSet<SingularContinuousVariable>(base.variables());
		baseVariables.removeAll(variables);

		removeNode(baseNode);

		ContinuousBeliefNode newReducedNode =
				addNode(new JointContinuousVariable(baseVariables));
		ContinuousBeliefNode newSeparatedNode =
				addNode(new JointContinuousVariable(variables));

		if (connectNewNodes) {
			for (DirectedNode parent : parents) {
				addEdge(newReducedNode, parent);
				addEdge(newSeparatedNode, parent);
			}

			for (DirectedNode child : children) {
				addEdge(child, newReducedNode);
				addEdge(child, newSeparatedNode);
			}
		}

		// the likelihoods are expired by the previous functions so it needs
		// not be called again.

		return new Pair<ContinuousBeliefNode, ContinuousBeliefNode>(
				newReducedNode, newSeparatedNode);
	}

	private void addNode(BeliefNode node) {

		if (containsNode(node.getName())) {
			System.out.println("node contained.");
		}
		// name must be unique in this BN. note that the name is unique implies
		// that the variable is unique, too. note also that the name of a
		// variable has already been trimmed.
		assert !containsNode(node.getName());

		// adds node to the list of nodes in this BN
		_nodes.add(node);

		// maps name to node
		putNode(node.getName(), node);

		// loglikelihoods expire
		expireLoglikelihoods();
	}

	/**
	 * Creates and returns a deep copy of this BN. This implementation copies
	 * everything in this BN but the name and variables. The default name will
	 * be used for the copy instead of the original one. The variables will be
	 * reused other than deeply copied. This will facilitate learning process.
	 * However, one cannot change node names after clone. TODO avoid redundant
	 * operations on CPTs.
	 * <p>
	 * Also note that cpts are also cloned.
	 * </p>
	 * 
	 * @return a deep copy of this BN.
	 */
	public BayesNet clone() {
		return new BayesNet(this);
	}

	/**
	 * Returns the standard dimension, namely, the number of free parameters in
	 * the CPTs, of this BN.
	 * 
	 * @return the standard dimension of this BN.
	 */
	public final int computeDimension() {
		// sums up dimension for each node
		int dimension = 0;

		for (AbstractNode node : _nodes) {
			dimension += ((BeliefNode) node).computeDimension();
		}

		return dimension;
	}

	/**
	 * <p>
	 * Makes loglikelihoods evaluated for this BN expired.
	 * </p>
	 * 
	 * <p>
	 * <b>Note: Besides methods in this class, only
	 * <code>BeliefNode.setCpt(Function)</code> is supposed to call this method.
	 * </p>
	 * 
	 * @see DiscreteBeliefNode#setCpt(Function)
	 */
	protected final void expireLoglikelihoods() {
		if (!_loglikelihoods.isEmpty()) {
			_loglikelihoods.clear();
		}
	}

	/**
	 * Get AICc Score of this BayesNet w.r.t data. Note that the loglikelihood
	 * of the model has already been computed and been stored in this model. (We
	 * should improve this point later.)
	 * 
	 * @author wangyi
	 * @param data
	 * @return AICc Score computed.
	 */
	public final double getAICcScore(DataSet data) {
		// TODO We will deal with the expiring case later.
		double logL = this.getLoglikelihood(data);

		assert logL != Double.NaN;

		// c.f. http://en.wikipedia.org/wiki/Akaike_information_criterion
		int k = computeDimension();
		return logL - k - k * (k + 1) / (data.getTotalWeight() - k - 1);
	}

	/**
	 * Get AIC Score of this BayesNet w.r.t data. Note that the loglikelihood of
	 * the model has already been computed and been stored in this model. (We
	 * should improve this point later.)
	 * 
	 * @author wangyi
	 * @param data
	 * @return AIC Score computed.
	 */
	public final double getAICScore(DataSet data) {
		// TODO We will deal with the expiring case later.
		double logL = this.getLoglikelihood(data);

		assert logL != Double.NaN;

		return logL - this.computeDimension();
	}

	/**
	 * Get BIC Score of this BayesNet w.r.t data. Note that the loglikelihood of
	 * the model has already been computed and been stored in this model.(We
	 * should improve this point later.)
	 * 
	 * @author csct Added by Chen Tao
	 * @param data
	 * @return BIC Score computed.
	 */
	public final double getBICScore(DataSet data) {

		// TODO We will deal with the expiring case later.
		double logL = this.getLoglikelihood(data);

		assert logL != Double.NaN;

		return logL - this.computeDimension() * Math.log(data.getTotalWeight())
				/ 2.0;
	}

	/**
	 * Returns the set of internal variables.
	 * 
	 * @return The set of internal variables.
	 */
	public final Set<DiscreteVariable> getInternalVars() {
		Set<DiscreteVariable> vars = new HashSet<DiscreteVariable>();
		for (DiscreteVariable var : getDiscreteVariables()) {
			if (!getNode(var).isLeaf())
				vars.add(var);
		}
		return vars;
	}

	/**
	 * Returns the set of leaf variables.
	 * 
	 * @return The set of leaf variables.
	 */
	public final Set<DiscreteVariable> getLeafVars() {
		Set<DiscreteVariable> vars = new HashSet<DiscreteVariable>();
		for (DiscreteVariable var : getDiscreteVariables()) {
			if (getNode(var).isLeaf())
				vars.add(var);
		}
		return vars;
	}

	/**
	 * Return the loglikelihood of this BN with respect to the specified data
	 * set.
	 * 
	 * @param dataSet
	 *            data set at request.
	 * @return the loglikelihood of this BN with respect to the specified data
	 *         set; return <code>Double.NaN</code> if the loglikelihood has not
	 *         been evaluated yet.
	 */
	public final double getLoglikelihood(DataSet dataSet) {
		Double loglikelihood = _loglikelihoods.get(dataSet);

		return loglikelihood == null ? Double.NaN : loglikelihood;
	}

	/**
	 * Returns the name of this BN.
	 * 
	 * @return the name of this BN.
	 */
	public final String getName() {
		return _name;
	}

	/**
	 * Gets a belief node from this network by name.
	 * 
	 * @param name
	 *            name of the target node
	 * @return node with the specified name, or null if not found
	 */
	public BeliefNode getNode(String name) {
		return (BeliefNode) super.getNode(name);
	}

	/**
	 * Returns a continuous belief node of the specified name. It throws an
	 * exception if the node of the given name is not a continuous node.
	 * 
	 * @param name
	 *            name of the continuous node
	 * @return a continuous belief node of the specified name
	 */
	public ContinuousBeliefNode getContinuousNode(String name) {
		return (ContinuousBeliefNode) super.getNode(name);
	}

	/**
	 * Returns a discrete belief node of the specified name. It throws an
	 * exception if the node of the given name is not a discrete node.
	 * 
	 * @param name
	 *            name of the discrete node
	 * @return a discrete belief node of the specified name
	 */
	public DiscreteBeliefNode getDiscreteNode(String name) {
		return (DiscreteBeliefNode) super.getNode(name);
	}

	/**
	 * Returns the node to which the specified variable is attached in this BN.
	 * 
	 * @param variable
	 *            variable attached to the node.
	 * @return the node to which the specified variable is attached; returns
	 *         <code>null</code> if none uses this variable.
	 */
	public final DiscreteBeliefNode getNode(DiscreteVariable variable) {
		return _variables.get(variable);
	}

	/**
	 * Returns the continuous belief node containing the given variable.
	 * 
	 * @param variable
	 *            variable of the node
	 * @return node containing the given variable, or {@code null} if it is not
	 *         found
	 */
	public ContinuousBeliefNode getNode(ContinuousVariable variable) {
		return _variables.get(variable);
	}

	/**
	 * Returns the belief node containing the given variable.
	 * 
	 * @param variable
	 *            variable of the node
	 * @return node containing the given variable, or {@code null} if it is not
	 *         found
	 */
	public BeliefNode getNode(Variable variable) {
		return _variables.get(variable);
	}

	/**
	 * Returns the list of variables in this BN. For the sake of efficiency,
	 * this implementation returns the reference to the private field. Make sure
	 * you understand this before using this method.
	 * 
	 * @return the list of variables in this BN.
	 */
	public final Set<DiscreteVariable> getDiscreteVariables() {

		Set<DiscreteVariable> vars = new HashSet<DiscreteVariable>();
		for (DiscreteVariable var : _variables.discreteMap().keySet()) {
			vars.add(var);
		}
		return vars;
		// return _variables.keySet();
	}

	/**
	 * Returns the set of all singular continuous variables.
	 * 
	 * @return set of all singular continuous variables
	 */
	public Set<SingularContinuousVariable> getSingularContinuousVariables() {
		return Collections.unmodifiableSet(_variables.continuousMap().keySet());
	}

	public Set<Variable> getVariables() {
		return _variables.keySet();
	}

	/**
	 * Whether a variable has a cardinality that allows a regular model. Calling
	 * it with continuous variable argument returns {@code true}.
	 * 
	 * @param variable
	 *            variable under check
	 * @return whether the variable has a regular cardinality
	 */
	public boolean hasRegularCardinality(Variable variable) {
		return variable.accept(new Variable.Visitor<Boolean>() {

			@Override
			public Boolean visit(DiscreteVariable variable) {
				DiscreteBeliefNode node = getNode(variable);
				if (node.isLeaf())
					return true;

				return variable.getCardinality() <= node.computeMaxPossibleCardInHLCM();
			}

			// since continuous node must be leaf node, it returns true

			@Override
			public Boolean visit(JointContinuousVariable variable) {
				return true;
			}

			@Override
			public Boolean visit(SingularContinuousVariable variable) {
				return true;
			}

		});
	}

	/**
	 * Loads the specified BIF file.
	 * 
	 * @param file
	 *            BIF file that defines a BN.
	 */
	private final void loadBif(String file) throws IOException {
		StreamTokenizer tokenizer = new StreamTokenizer(new FileReader(file));

		tokenizer.resetSyntax();

		// characters that will be ignored
		tokenizer.whitespaceChars('=', '=');
		tokenizer.whitespaceChars(' ', ' ');
		tokenizer.whitespaceChars('"', '"');
		tokenizer.whitespaceChars('\t', '\t');

		// word characters
		tokenizer.wordChars('A', 'z');

		// we will parse numbers
		tokenizer.parseNumbers();

		// special characters considered in the gramma
		tokenizer.ordinaryChar(';');
		tokenizer.ordinaryChar('(');
		tokenizer.ordinaryChar(')');
		tokenizer.ordinaryChar('{');
		tokenizer.ordinaryChar('}');
		tokenizer.ordinaryChar('[');
		tokenizer.ordinaryChar(']');

		// does NOT treat eol as a token
		tokenizer.eolIsSignificant(false);

		// ignores c++ comments
		tokenizer.slashSlashComments(true);

		// starts parsing
		int value;

		// reads until the end of the stream (file)
		do {
			value = tokenizer.nextToken();

			if (value == StreamTokenizer.TT_WORD) {
				// start of a new block here
				String word = tokenizer.sval;

				if (word.equals("network")) {
					// parses network properties. next string must be the name
					// of this BN
					tokenizer.nextToken();
					setName(tokenizer.sval);
				} else if (word.equals("variable")) {
					// parses variable. get name of variable first
					tokenizer.nextToken();
					String name = tokenizer.sval;

					// looks for '['
					do {
						value = tokenizer.nextToken();
					} while (value != '[');

					// gets integer as cardinality
					tokenizer.nextToken();
					int cardinality = (int) tokenizer.nval;

					// looks for '{'
					do {
						value = tokenizer.nextToken();
					} while (value != '{');

					// state list
					ArrayList<String> states = new ArrayList<String>();

					// gets states
					do {
						value = tokenizer.nextToken();
						if (value == StreamTokenizer.TT_WORD) {
							states.add(tokenizer.sval);
						}
					} while (value != '}');

					// tests consistency
					assert states.size() == cardinality;

					// creates node
					addNode(new DiscreteVariable(name, states));
				} else if (word.equals("probability")) {
					// parses CPT. skips next '('
					tokenizer.nextToken();

					// variables in this family
					ArrayList<DiscreteVariable> family =
							new ArrayList<DiscreteVariable>();

					// gets variable name and node
					tokenizer.nextToken();
					DiscreteBeliefNode node =
							(DiscreteBeliefNode) getNode(tokenizer.sval);
					family.add(node.getVariable());

					// gets parents and adds edges
					do {
						value = tokenizer.nextToken();
						if (value == StreamTokenizer.TT_WORD) {
							DiscreteBeliefNode parent =
									(DiscreteBeliefNode) getNode(tokenizer.sval);
							family.add(parent.getVariable());

							// adds edge from parent to node
							addEdge(node, parent);
						}
					} while (value != ')');

					// creates CPT
					Function cpt = Function.createFunction(family);

					// looks for '(' or words
					do {
						value = tokenizer.nextToken();
					} while (value != '(' && value != StreamTokenizer.TT_WORD);

					// checks next token: there are two formats, one with
					// "table" and the other fills in cells one by one.
					if (value == StreamTokenizer.TT_WORD) {
						// we only accept "table" but not "default"
						assert tokenizer.sval.equals("table");

						// probability values
						ArrayList<Double> values = new ArrayList<Double>();

						// gets numerical tokens
						do {
							value = tokenizer.nextToken();
							if (value == StreamTokenizer.TT_NUMBER) {
								values.add(tokenizer.nval);
							}
						} while (value != ';');

						// consistency between family and values will be tested
						cpt.setCells(family, values);
					} else {
						// states array
						ArrayList<Integer> states = new ArrayList<Integer>();
						states.add(0);
						int cardinality = node.getVariable().getCardinality();

						// parses row by row
						while (value != '}') {
							// gets parent states
							for (int i = 1; i < family.size(); i++) {
								do {
									value = tokenizer.nextToken();
								} while (value != StreamTokenizer.TT_WORD);
								states.add(family.get(i).indexOf(tokenizer.sval));
							}

							// fills in data
							for (int i = 0; i < cardinality; i++) {
								states.set(0, i);

								do {
									value = tokenizer.nextToken();
								} while (value != StreamTokenizer.TT_NUMBER);
								cpt.setCell(family, states, tokenizer.nval);
							}

							// looks for next '(' or '}'
							while (value != '(' && value != '}') {
								value = tokenizer.nextToken();
							}
						}
					}

					// normalizes the CPT with respect to the attached variable
					cpt.normalize(node.getVariable());

					// sets the CPT
					node.setCpt(cpt);
				}
			}
		} while (value != StreamTokenizer.TT_EOF);
	}

	/**
	 * Randomly sets the parameters of this BN. TODO avoid redundant operations
	 * on CPTs.
	 */
	public final void randomlyParameterize() {
		for (AbstractNode node : _nodes) {
			((DiscreteBeliefNode) node).randomlyParameterize();
		}

		// loglikelihoods expire
		expireLoglikelihoods();
	}

	/**
	 * Randomly sets the parameters of the specified list of nodes in this BN.
	 * TODO avoid redundant operations on CPTs.
	 * 
	 * @param mutableNodes
	 *            list of nodes whose parameters are to be randomized.
	 */
	public final void randomlyParameterize(
			Collection<DiscreteBeliefNode> mutableNodes) {
		// mutable nodes must be in this BN
		assert _nodes.containsAll(mutableNodes);

		for (DiscreteBeliefNode node : mutableNodes) {
			node.randomlyParameterize();
		}

		// loglikelihoods expire
		expireLoglikelihoods();
	}

	/**
	 * Removes the specified edge from this BN. This implementation extends
	 * <code>AbstractGraph.removeEdge(Edge)</code> such that all loglikelihoods
	 * will be expired.
	 * 
	 * @param edge
	 *            edge to be removed from this BN.
	 */
	@Override
	public final void removeEdge(Edge edge) {
		super.removeEdge(edge);

		// loglikelihoods expire
		expireLoglikelihoods();
	}

	/**
	 * Removes the specified node from this BN. This implementation extends
	 * <code>AbstractGraph.removeNode(AbstractNode)</code> such that map from
	 * variables to nodes will be updated and all loglikelihoods will be
	 * expired.
	 * 
	 * @param node
	 *            node to be removed from this BN.
	 */
	public final void removeNode(AbstractNode node) {
		super.removeNode(node);
		_variables.remove(((BeliefNode) node).getVariable());

		// loglikelihoods expire
		expireLoglikelihoods();
	}

	/**
	 * Generates a batch of samples from this BN.
	 * 
	 * @param sampleSize
	 *            number of samples to be generated.
	 * @return a batch of samples from this BN.
	 */
	public DataSet sample(int sampleSize) {
		// initialize data set
		int nNodes = getNumberOfNodes();
		DataSet samples =
				new DataSet(getDiscreteVariables().toArray(
						new DiscreteVariable[nNodes]));

		// since variables are sorted in data set, find mapping from belief
		// nodes to variables
		DiscreteVariable[] vars = samples.getVariables();
		HashMap<AbstractNode, Integer> map =
				new HashMap<AbstractNode, Integer>();
		for (AbstractNode node : getNodes()) {
			DiscreteVariable var = ((DiscreteBeliefNode) node).getVariable();
			int pos = Arrays.binarySearch(vars, var);
			map.put(node, pos);
		}

		// topological sort
		AbstractNode[] order = topologicalSort();

		for (int i = 0; i < sampleSize; i++) {
			int[] states = new int[nNodes];

			// forward sampling
			for (AbstractNode node : order) {
				DiscreteBeliefNode bNode = (DiscreteBeliefNode) node;

				// find parents and their states
				ArrayList<DiscreteVariable> parents =
						new ArrayList<DiscreteVariable>();
				ArrayList<Integer> parentStates = new ArrayList<Integer>();
				for (DirectedNode parent : bNode.getParents()) {
					DiscreteVariable var =
							((DiscreteBeliefNode) parent).getVariable();
					parents.add(var);

					int pos = map.get(parent);
					parentStates.add(states[pos]);
				}

				// instantiate parents
				Function cond =
						bNode.potential().project(parents, parentStates);

				// sample according to the conditional distribution
				states[map.get(node)] = cond.sample();
			}

			// add to samples
			samples.addDataCase(states, 1.0);
		}

		return samples;
	}

	/**
	 * Outputs this BN to the specified file in BIF format. See
	 * http://www.cs.cmu
	 * .edu/~fgcozman/Research/InterchangeFormat/Old/xmlbif02.html for the
	 * grammar of BIF format.
	 * 
	 * @param file
	 *            output of this BN.
	 * @throws FileNotFoundException
	 *             if the file exists but is a directory rather than a regular
	 *             file, does not exist but cannot be created, or cannot be
	 *             opened for any other reason.
	 */
	public final void saveAsBif(String file) throws FileNotFoundException {
		PrintWriter out = new PrintWriter(file);

		// outputs header
		out.println("// " + file);
		out.println("// Produced by org.latlab at "
				+ (new Date(System.currentTimeMillis())));

		// outputs name
		out.println("network \"" + _name + "\" {");
		out.println("}");
		out.println();

		// outputs nodes
		for (AbstractNode node : _nodes) {
			DiscreteVariable variable =
					((DiscreteBeliefNode) node).getVariable();

			// name of variable
			out.println("variable \"" + variable.getName() + "\" {");

			// states of variable
			out.print("\ttype discrete[" + variable.getCardinality() + "] { ");
			Iterator<String> iter = variable.getStates().iterator();
			while (iter.hasNext()) {
				out.print("\"" + iter.next() + "\"");
				if (iter.hasNext()) {
					out.print(" ");
				}
			}
			out.println(" };");

			out.println("}");
			out.println();
		}

		// Output CPTs
		for (AbstractNode node : _nodes) {
			DiscreteBeliefNode bNode = (DiscreteBeliefNode) node;

			// variables in this family. note that the variables in the
			// probability block are arranged from the most significant place to
			// the least significant place.
			ArrayList<DiscreteVariable> vars =
					new ArrayList<DiscreteVariable>();
			vars.add(bNode.getVariable());

			// name of node
			out.print("probability ( \"" + bNode.getName() + "\" ");

			// names of parents
			if (!bNode.isRoot()) {
				out.print("| ");
			}

			Iterator<DirectedNode> iter = bNode.getParents().iterator();
			while (iter.hasNext()) {
				DiscreteBeliefNode parent = (DiscreteBeliefNode) iter.next();

				out.print("\"" + parent.getName() + "\"");
				if (iter.hasNext()) {
					out.print(", ");
				}

				vars.add(parent.getVariable());
			}
			out.println(" ) {");

			// cells in CPT
			out.print("\ttable");
			for (double cell : bNode.potential().getCells(vars)) {
				out.print(" " + cell);
			}
			out.println(";");
			out.println("}");
		}

		out.close();
	}

	/**
	 * Replaces the loglikelihood of this BN with respect to the specified data
	 * set.
	 * 
	 * @param dataSet
	 *            data set at request.
	 * @param loglikelihood
	 *            new loglikelihood of this BN.
	 */
	public final void setLoglikelihood(DataSet dataSet, double loglikelihood) {
		// loglikelihood must be non-positive
		assert loglikelihood <= 0.0;

		_loglikelihoods.put(dataSet, loglikelihood);
	}

	/**
	 * Replaces the name of this BN.
	 * 
	 * @param name
	 *            new name of this BN.
	 */
	public final void setName(String name) {
		name = name.trim();

		// name cannot be blank
		assert name.length() > 0;

		_name = name;
	}

	/**
	 * Returns a string representation of this BN. The string representation
	 * will be indented by the specified amount.
	 * 
	 * @param amount
	 *            amount by which the string representation is to be indented.
	 * @return a string representation of this BN.
	 */
	public String toString(int amount) {
		// amount must be non-negative
		assert amount >= 0;

		// prepares white space for indent
		StringBuffer whiteSpace = new StringBuffer();
		for (int i = 0; i < amount; i++) {
			whiteSpace.append('\t');
		}

		// builds string representation
		StringBuffer stringBuffer = new StringBuffer();

		stringBuffer.append(whiteSpace);
		stringBuffer.append(getName() + " {\n");

		stringBuffer.append(whiteSpace);
		stringBuffer.append("\tnumber of nodes = " + getNumberOfNodes() + ";\n");

		stringBuffer.append(whiteSpace);
		stringBuffer.append("\tnodes = {\n");

		for (AbstractNode node : _nodes) {
			stringBuffer.append(node.toString(amount + 2));
		}

		stringBuffer.append(whiteSpace);
		stringBuffer.append("\t};\n");

		stringBuffer.append(whiteSpace);
		stringBuffer.append("\tnumber of edges = " + getNumberOfEdges() + ";\n");

		stringBuffer.append(whiteSpace);
		stringBuffer.append("\tedges = {\n");

		for (Edge edge : _edges) {
			stringBuffer.append(edge.toString(amount + 2));
		}

		stringBuffer.append(whiteSpace);
		stringBuffer.append("\t};\n");

		stringBuffer.append(whiteSpace);
		stringBuffer.append("};\n");

		return stringBuffer.toString();
	}

	/**
	 * Finds a variable by name. It supports only singular variable (continuous
	 * or discrete).
	 * 
	 * @param name
	 *            name of the variable
	 * @return variable with the given name, or {@code null} if not found
	 */
	public Variable findVariableByName(String name) {
		return _variables.findVariableByName(name);
	}

	/**
	 * Finds a joint continuous variable by the name of one of its singular
	 * variable.
	 * 
	 * @param singularVariableName
	 *            name of the singular continuous variable
	 * @return joint variable containing the given singular variable
	 */
	public JointContinuousVariable findJointVariableByName(
			String singularVariableName) {
		SingularContinuousVariable variable =
				(SingularContinuousVariable) _variables.findVariableByName(singularVariableName);
		ContinuousBeliefNode node = getNode(variable);
		return node != null ? node.getVariable() : null;
	}
}
