package org.latlab.core.util;

/**
 * Generates hash code according to Effective Java.
 * 
 * <p>
 * To generate a hash code for an instance:
 * <ol>
 * <li>Construct a new generator or call {@link #reset()}.
 * <li>For each field used in the {@code equal} comparison, calls
 * {@code addField} on that field.
 * <li>Calls {@link #current()} to get the generated hash code.
 * </ol>
 * 
 * @author leonard
 * 
 */
public class HashCodeGenerator {
    private int result;

    public HashCodeGenerator() {
        reset();
    }

    public int current() {
        return result;
    }

    /**
     * Resets the generator.
     */
    public void reset() {
        result = 17;
    }

    public void addField(boolean value) {
        combine(value ? 0 : 1);
    }

    public void addField(int value) {
        combine(value);
    }

    public void addField(long value) {
        combine((int) (value ^ (value >>> 32)));
    }

    public void addField(float value) {
        addField(Float.floatToIntBits(value));
    }

    public void addField(double value) {
        addField(Double.doubleToLongBits(value));
    }

    public void addField(Object o) {
        addField(o == null ? 0 : o.hashCode());
    }
    
    public void addHashCode(int hashCode) {
        combine(hashCode);
    }

    private void combine(int c) {
        result = 37 * result + c;
    }
}
