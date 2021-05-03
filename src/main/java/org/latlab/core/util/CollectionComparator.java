package org.latlab.core.util;

import java.util.Collection;
import java.util.Comparator;
import java.util.Iterator;

/**
 * Compares two collections of comparable items in dictionary order. The shorter
 * are treated as "less than" the longer collection.
 * 
 * @author leonard
 * 
 * @param <T>
 */
public class CollectionComparator<T extends Comparable<? super T>>
    implements Comparator<Collection<T>> {

    public int compare(Collection<T> o1, Collection<T> o2) {
        Iterator<T> i1 = o1.iterator();
        Iterator<T> i2 = o2.iterator();
        
        while (i1.hasNext()) {
            if (i2.hasNext()) {
                int result = i1.next().compareTo(i2.next());
                if (result != 0)
                    return result;
            } else {
                return 1;
            }
        }
        
        return i2.hasNext()? -1 : 0; 
    }
}
