package org.latlab.core.model;

import org.latlab.core.util.Converter;
import org.latlab.core.util.Variable;

/**
 * Extracts the variable from a belief node.
 * 
 * @author leonard
 * 
 */
public class VariableExtractor implements Converter<BeliefNode, Variable> {

    /**
     * Converts a belief node to its variable.
     * 
     * @param node
     *            node to convert from
     * @return variable of the belief node
     */
    public Variable convert(BeliefNode node) {
        return node.getVariable();
    }

}
