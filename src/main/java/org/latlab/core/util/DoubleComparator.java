package org.latlab.core.util;

/**
 * Compares the double with the consideration that the double arguments may be
 * {@code Double.NaN}. It reverses the normal behavior of {@code Double.compare}
 * , and treats the {@code Double.NaN} as a smaller value.
 * 
 * @author leonard
 * 
 */
public class DoubleComparator {
    public static int compare(double d1, double d2) {
        if (!Double.isNaN(d1) && !Double.isNaN(d2)) {
            // both are not NaN
            return Double.compare(d1, d2);
        }
        else {
            // at least one of them is NaN
            return -Double.compare(d1, d2);
        }
    }
}
