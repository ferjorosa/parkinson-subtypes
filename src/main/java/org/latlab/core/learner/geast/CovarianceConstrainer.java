package org.latlab.core.learner.geast;

import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.colt.matrix.linalg.EigenvalueDecomposition;
import org.latlab.core.graph.AbstractNode;
import org.latlab.core.model.BayesNet;
import org.latlab.core.model.CGParameter;
import org.latlab.core.model.CGPotential;
import org.latlab.core.model.ContinuousBeliefNode;
import org.latlab.core.util.SingularContinuousVariable;

import java.io.PrintWriter;
import java.util.Collection;

/**
 * Detects whether an estimation has found a spurious local maximizer.
 * 
 * @author leonard
 * 
 */
public abstract class CovarianceConstrainer {

	@SuppressWarnings("serial")
	public static class ImproperValueException extends RuntimeException {
		public ImproperValueException(Throwable cause) {
			super(cause);
		}
	}

	protected final Algebra algebra = Algebra.DEFAULT;
	protected final double tolerance = algebra.property().tolerance();

	protected abstract double getLowerBound(
			Collection<SingularContinuousVariable> variables);

	protected abstract double getUpperBound(
			Collection<SingularContinuousVariable> variables);

	public boolean detect(BayesNet model) {
		for (AbstractNode node : model.getNodes()) {
			if (node instanceof ContinuousBeliefNode) {
				if (isPotentialOutOfBounds((ContinuousBeliefNode) node))
					return true;
			}
		}
		return false;
	}

	protected boolean isPotentialOutOfBounds(ContinuousBeliefNode node) {
		Collection<SingularContinuousVariable> variables =
				node.getVariable().variables();
		double lower = getLowerBound(variables) - tolerance;
		double upper = getUpperBound(variables) + tolerance;

		CGPotential potential = node.potential();

		for (int i = 0; i < potential.size(); i++) {
			CGParameter parameter = potential.get(i);

			EigenvalueDecomposition d = null;
			try {
				d = new EigenvalueDecomposition(parameter.C);
			} catch (Exception e) {
				throw new ImproperValueException(e);
			}

			DoubleMatrix2D eigenvalues = d.getD();

			// check whether any of the eigenvalues fall outside the bounds,
			// and adjust them if necessary
			for (int j = 0; j < eigenvalues.rows(); j++) {
				double value = eigenvalues.getQuick(j, j);
				if (value < lower || value > upper)
					return true;
			}
		}

		return false;
	}

	/**
	 * Adjusts the potential if any of the parameters is deemed improper by this
	 * corrector.
	 * 
	 * @param potential
	 */
	public void adjust(CGPotential potential) {
		Collection<SingularContinuousVariable> variables =
				potential.continuousVariables();
		double lower = getLowerBound(variables);
		double upper = getUpperBound(variables);

		for (int i = 0; i < potential.size(); i++) {
			CGParameter parameter = potential.get(i);

			EigenvalueDecomposition d = null;
			try {
				d = new EigenvalueDecomposition(parameter.C);
			} catch (Exception e) {
				// in some rare case the decomposition fails and throws an
				// ArrayOutOfBoundsException
				throw new ImproperValueException(e);
			}

			DoubleMatrix2D eigenvalues = d.getD();
			boolean changed = false;

			// check whether any of the eigenvalues fall outside the bounds,
			// and adjust them if necessary
			for (int j = 0; j < eigenvalues.rows(); j++) {
				double value = eigenvalues.getQuick(j, j);
				if (value < lower - tolerance) {
					eigenvalues.setQuick(j, j, lower);
					changed = true;
				} else if (value > upper + tolerance) {
					eigenvalues.setQuick(j, j, upper);
					changed = true;
				}
			}

			// if the eigenvalues have been adjusted, compute the covariance
			// matrix
			if (changed) {
				DoubleMatrix2D v = d.getV();
				DoubleMatrix2D vt = algebra.transpose(v);
				DoubleMatrix2D m1 = algebra.mult(v, eigenvalues);
				parameter.C = algebra.mult(m1, vt);
			}
		}
	}

	public abstract void writeXml(PrintWriter writer);
}
