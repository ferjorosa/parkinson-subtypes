package org.latlab.core.util;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

/**
 * Generates different combinations out of a given collection of items.
 * 
 * @author leonard
 * 
 * @param <T>
 *            type of items
 */
public class CombinationGenerator<T extends Comparable<? super T>> {
    private final List<T> items;

    private class Generator {
        private final int size;
        private final List<T> current;
        private final List<List<T>> all;

        public Generator(int size) {
            this.size = size;
            current = new ArrayList<T>(size);
            all = new LinkedList<List<T>>();
        }

        public List<List<T>> generate() {
            all.clear();

            if (size < 1)
                return all;

            generate(0);

            return all;
        }

        /**
         * This adds a different item to the current list until it reaches its
         * desired size.
         * 
         * @param start
         *            the first item to consider next
         */
        private void generate(int start) {
            if (current.size() == size) {
                all.add(new ArrayList<T>(current));
                return;
            }

            for (int i = start; i < items.size(); i++) {
                // try different item at this level and then pass it to the next
                // level
                current.add(items.get(i));
                generate(i + 1);
                current.remove(current.size() - 1);
            }
        }
    }

    public CombinationGenerator(List<T> items) {
        this.items = items;
    }

    public CombinationGenerator(Collection<T> items) {
        this.items = new ArrayList<T>(items);
    }

    public List<List<T>> generate(int size) {
        return new Generator(size).generate();
    }
}
