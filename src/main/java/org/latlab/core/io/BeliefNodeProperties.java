package org.latlab.core.io;

import org.latlab.core.model.BeliefNode;

import java.util.HashMap;

/**
 * A map holding the properties of all belief nodes.  
 * The belief nodes are used as the keys.
 * @author leonard
 *
 */
public class BeliefNodeProperties 
	extends HashMap<BeliefNode, BeliefNodeProperty> {

	private static final long serialVersionUID = 2285807628705084093L;
}
