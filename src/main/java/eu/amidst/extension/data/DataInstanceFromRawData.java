package eu.amidst.extension.data;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.Attributes;
import eu.amidst.core.datastream.DataInstance;

import java.util.Arrays;

/**
 * Implementacion propia de DataInstance para cuando proyectamos o completamos datos
 */
public class DataInstanceFromRawData implements DataInstance {

    private Attributes attributes;

    private double[] data;

    public DataInstanceFromRawData(Attributes attributes, double[] data) {
        this.attributes = attributes;
        this.data = data;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double getValue(Attribute att) {
        return data[att.getIndex()];
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setValue(Attribute att, double value) {
        this.data[att.getIndex()]=value;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Attributes getAttributes() {
        return this.attributes;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double[] toArray() {
        return this.data;
    }

    @Override
    public String toString() {
        return Arrays.toString(data);
    }
}
