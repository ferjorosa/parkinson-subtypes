package org.latlab.core.util;

/**
 * A pair of two elements, which can be of same or different types.
 * @author leonard
 *
 * @param <T1>  first type of element
 * @param <T2>  second type of element
 */
public class Pair<T1, T2> {
    public Pair() {
        this(null, null);
    }
    
    public Pair(T1 o1, T2 o2) {
        first = o1;
        second = o2;
    }
    
    public static <T1, T2> Pair<T1, T2> construct(T1 o1, T2 o2) {
    	return new Pair<T1, T2>(o1, o2);
    }
    
    @SuppressWarnings("unchecked")
    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        
        if (!(obj instanceof Pair)) {
            return false;
        }
        
        Pair pair = (Pair) obj;
        
        return first == pair.first ||
                (first != null && first.equals(pair.first)) &&
        (second == pair.second || 
                (second != null && second.equals(pair.second)));
    }

    @Override
    public int hashCode() {
        int result = 17;
        result = 37 * result + first.hashCode();
        result = 37 * result + second.hashCode();
        return result;
    }

    public final T1 first;
    public final T2 second;
}
