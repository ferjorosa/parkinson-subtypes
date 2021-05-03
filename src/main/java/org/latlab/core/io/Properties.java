package org.latlab.core.io;

import org.latlab.core.model.BeliefNode;
import org.latlab.core.model.ContinuousBeliefNode;
import org.latlab.core.model.DiscreteBeliefNode;

import java.util.List;

/**
 * Holds the properties of a Bayesian network, including those of nodes and
 * probability definitions.
 * 
 * It is responsible to read the property strings found in the the network file
 * and generate those strings for writing to the file.
 * 
 * @author leonard
 * 
 */
public class Properties {
	public void readNetworkProperty(String value) {

	}

	public void readVariableProperty(DiscreteBeliefNode node, String value) {

	}

	/**
	 * Gets the property of a belief node, and creates a new property if it does
	 * not exist for the node.
	 * 
	 * @param node
	 *            node whose property is returned
	 * @return property of the specified node
	 */
	public BeliefNodeProperty getBeliefNodeProperty(BeliefNode node) {
		BeliefNodeProperty property = nodeProperties.get(node);

		if (property == null) {
			property = new BeliefNodeProperty();
			nodeProperties.put(node, property);
		}

		return property;
	}

	/**
	 * Called when some continuous nodes are combined into one node. The
	 * property of the first constituent node will become the property of the
	 * combined node.
	 * 
	 * @param combined
	 *            combined node
	 * @param constituents
	 *            constituent nodes
	 */
	public void combine(ContinuousBeliefNode combined,
			List<ContinuousBeliefNode> constituents) {
		BeliefNodeProperty property = null;
		for (ContinuousBeliefNode node : constituents) {
			if (property == null) {
				property = nodeProperties.remove(node);
			} else {
				nodeProperties.remove(node);
			}
		}
		
		if (property != null) {
			nodeProperties.put(combined, property);
		}
	}

	public BeliefNodeProperties getBeliefNodeProperties() {
		return nodeProperties;
	}

	// private final NetworkProperty networkProperty = new NetworkProperty();
	private final BeliefNodeProperties nodeProperties =
			new BeliefNodeProperties();
}
