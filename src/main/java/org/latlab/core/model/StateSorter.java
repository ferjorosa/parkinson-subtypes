package org.latlab.core.model;

import org.latlab.core.graph.AbstractNode;
import org.latlab.core.graph.Edge;
import org.latlab.core.graph.search.AbstractVisitor;
import org.latlab.core.graph.search.DepthFirstSearch;
import org.latlab.core.graph.search.Visitor;
import org.latlab.core.util.ComparablePair;
import org.latlab.core.util.Function;

import java.util.*;

public class StateSorter {
    public static void sort(Gltm model) {
        DepthFirstSearch search = new DepthFirstSearch(model);
        search.perform(model.getRoot(), createVisitor());
    }

    private static Visitor createVisitor() {
        return new AbstractVisitor() {

            private Set<AbstractNode> visited = new HashSet<AbstractNode>();

            public boolean discover(AbstractNode node, Edge edge) {
                if (visited.contains(node))
                    return false;

                if (node instanceof DiscreteBeliefNode)
                    order((DiscreteBeliefNode) node);

                return true;
            }

            public void finish(AbstractNode node) {}

            private void order(DiscreteBeliefNode node) {
                Function potential =
                    node.potential().marginalize(node.getVariable());

                double[] cells = potential.getCells();

                List<ComparablePair<Double, Integer>> list =
                    new ArrayList<ComparablePair<Double, Integer>>(cells.length);

                for (int i = 0; i < cells.length; i++) {
                    list.add(new ComparablePair<Double, Integer>(cells[i], i));
                }

                Collections.sort(list);

                int[] order = new int[list.size()];
                for (int i = 0; i < list.size(); i++) {
                    order[i] = list.get(i).second;
                }

                node.reorderStates(order);
            }

        };
    }
}
