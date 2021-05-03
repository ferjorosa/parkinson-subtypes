package org.latlab.core.graph.search;

import org.latlab.core.graph.AbstractNode;
import org.latlab.core.graph.Edge;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * A visitor to record the discovering and finishing times
 * of the nodes during a depth first search.
 * @author leonard
 *
 */
public class TimeVisitor extends AbstractVisitor {
    
	public TimeVisitor() {
		time = 0;
	}
    /**
     * Constructor
     * @param initialTime   initial time
     */
    public TimeVisitor(int initialTime) {
        time = initialTime;
    }

    public boolean discover(AbstractNode node, Edge edge) {
    	if (discoveringTimes.containsKey(node) ||
    			ignoredEdges.contains(edge))
    		return false;
    	
        discoveringTimes.put(node, time++);
        return true;
    }
    
    public void finish(AbstractNode node) {
        finishingTimes.put(node, time++);
    }
    
    public boolean discovered(AbstractNode node) {
        return discoveringTimes.containsKey(node);
    }
    
    /**
     * Indicates an edge that should be ignored during the search.
     * @param edge	an edge that should be ignored
     */
    public void addIgnoredEdge(Edge edge) {
    	ignoredEdges.add(edge);
    }
    
    public final Map<AbstractNode, Integer> discoveringTimes =
        new HashMap<AbstractNode, Integer>();
    public final Map<AbstractNode, Integer> finishingTimes =
        new HashMap<AbstractNode, Integer>();
    
    private int time = 0;
    private Set<Edge> ignoredEdges = new HashSet<Edge>();
}
