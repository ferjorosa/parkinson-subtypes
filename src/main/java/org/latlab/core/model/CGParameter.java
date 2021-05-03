package org.latlab.core.model;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import org.latlab.core.util.MixtureOfGaussianStructure;

/**
 * A entry corresponding to one combination of parent discrete variables that
 * makes up the CG potential.
 * 
 * @author leonard
 * 
 */
public class CGParameter extends MixtureOfGaussianStructure {

	public CGParameter(int numberOfHeadVariables) {
	    super(numberOfHeadVariables, 1);
	    
	    // initialize the variance to a non-zero number
	    setDiagonalTo(1);
	}
	
	public CGParameter(double p, DoubleMatrix1D A, DoubleMatrix2D C) {
	    super(p, A, C);
	}
	
	private CGParameter(CGParameter parameter) {
	    super(parameter);
	}

	public CGParameter copy() {
		return new CGParameter(this);
	}
	
	private void setDiagonalTo(double value) {
	    for (int i = 0; i < C.columns(); i++) {
	        C.setQuick(i, i, value);
	    }
	}
}
