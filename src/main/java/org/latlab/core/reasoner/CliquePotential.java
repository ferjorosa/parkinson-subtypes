package org.latlab.core.reasoner;


public class CliquePotential<T> {
	public final T content;
	public double logNormalization = 0;
	
	public CliquePotential(T c) {
		content = c;
	}
	
	public CliquePotential(T c, double v) {
		content = c;
		logNormalization = v;
	}
}
