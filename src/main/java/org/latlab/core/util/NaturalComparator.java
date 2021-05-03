package org.latlab.core.util;

import java.util.Comparator;

/**
 * A {@code comparator} using the natural order of a particular type.
 * 
 * @author leonard
 * 
 * @param <T>
 *            type of elements being compared
 */
public class NaturalComparator<T extends Comparable<T>>
    implements Comparator<T> {

    public int compare(T o1, T o2) {
        return o1.compareTo(o2);
    }
}
