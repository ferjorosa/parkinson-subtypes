package org.latlab.core.io;

import org.latlab.core.model.BayesNet;

/**
 * Parser interface.
 * 
 * @author leonard
 * 
 */
public interface Parser {
	// /**
	// * Parses the embedded input stream and
	// * replaces the network content with that in the stream
	// * @param network network to hold the persisted content
	// */
	// void parse(BayesNet network);

	/**
	 * Parses the embedded input stream and returns the network read from the
	 * stream
	 * 
	 * @return network parsed from input stream
	 */
	public <T extends BayesNet> T parse(T network) throws ParseException;

	/**
	 * Gets the properties found in the network file just read.
	 * 
	 * @return Properties of network, variables, probability definitions.
	 */
	Properties getProperties();
}
