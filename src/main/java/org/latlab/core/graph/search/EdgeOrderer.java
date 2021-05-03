package org.latlab.core.graph.search;

import org.latlab.core.graph.AbstractNode;
import org.latlab.core.graph.Edge;

import java.util.Collection;

/**
 * Gives an ordering to those alternatives edges so that the search 
 * explores the edges following this ordering. 
 * @author leonard
 *
 */
public interface EdgeOrderer {
    public Collection<Edge> order(AbstractNode current, Collection<Edge> edges);
}
