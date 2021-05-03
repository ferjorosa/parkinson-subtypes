package org.latlab.core.util;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

import java.util.ArrayList;
import java.util.List;

/**
 * A structure for holding numbers related to a mixture of Gaussian distribution
 * 
 * @author leonard
 * 
 */
public class MixtureOfGaussianStructure {
    public double p;
    public DoubleMatrix1D A;
    public DoubleMatrix2D C;

    protected MixtureOfGaussianStructure(int variables, double p) {
        this(p, new DenseDoubleMatrix1D(variables), new DenseDoubleMatrix2D(
            variables, variables));
    }

    public MixtureOfGaussianStructure(
            double p, DoubleMatrix1D A, DoubleMatrix2D C) {
        this.p = p;
        this.A = A;
        this.C = C;
    }

    protected MixtureOfGaussianStructure(MixtureOfGaussianStructure structure) {
        this(structure.p, structure.A.copy(), structure.C.copy());
    }

    public MixtureOfGaussianStructure copy() {
        return new MixtureOfGaussianStructure(this);
    }

    /**
     * Returns a serialized representation of the mean vector and covariance
     * matrix.
     * 
     * @return a serialized representation
     */
    public List<Double> getEntries() {
        List<Double> entries = new ArrayList<Double>(A.size() + C.size());
        for (int i = 0; i < A.size(); i++) {
            entries.add(A.get(i));
        }

        // set the covariance matrix
        for (int row = 0; row < C.rows(); row++) {
            for (int column = 0; column < C.columns(); column++) {
                entries.add(C.get(row, column));
            }
        }

        return entries;
    }

    /**
     * Sets the values of this parameter.
     * 
     * @param entries
     *            entries for setting values
     */
    public void setEntries(List<Double> entries) {
        int index = 0;

        // set the mean vector
        for (int i = 0; i < A.size(); i++) {
            A.set(i, entries.get(index));
            index++;
        }

        // set the covariance matrix
        for (int row = 0; row < C.rows(); row++) {
            for (int column = 0; column < C.columns(); column++) {
                C.set(row, column, entries.get(index));
                index++;
            }
        }
    }

    
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append(getClass().getSimpleName() + "\n");
        builder.append(p);
        builder.append("\n");
        builder.append(A);
        builder.append("\n");
        builder.append(C);
        builder.append("\n");
        
        return builder.toString();
    }
    
    
    @Override
    public boolean equals(Object o) {
        if (o == this)
            return true;
        else if (o == null)
            return false;
        else if (o instanceof MixtureOfGaussianStructure) {
            MixtureOfGaussianStructure other = (MixtureOfGaussianStructure) o;
            return p == other.p && A.equals(other.A) && C.equals(other.C);
        } else
            return false;
    }

    @Override
    public int hashCode() {
        HashCodeGenerator generator = new HashCodeGenerator();
        generator.addField(p);
        generator.addField(A);
        generator.addField(C);
        return generator.current();
    }
}
