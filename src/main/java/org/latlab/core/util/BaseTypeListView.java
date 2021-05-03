package org.latlab.core.util;

import java.util.AbstractList;
import java.util.List;

/**
 * A immutable view of a base type of another list. 
 * @author leonard
 *
 * @param <T> type of the elements in this view
 */
public class BaseTypeListView<T> extends AbstractList<T> {
    private final List<? extends T> original;

    public BaseTypeListView(List<? extends T> list) {
        original = list;
    }
    
    @Override
    public T get(int index) {
        return original.get(index);
    }

    @Override
    public int size() {
        return original.size();
    }

}
