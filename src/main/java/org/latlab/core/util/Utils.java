/**
 * Utils.java Copyright (C) 2007 Tao Chen, Kin Man Poon, Yi Wang, and Nevin L.
 * Zhang
 */
package org.latlab.core.util;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;

public class Utils {

	/**
	 * Returns the conditional mutual information I(X;Y|Z) given the 3D
	 * distribution P(X,Y,Z).
	 * 
	 * @param dist
	 *            Distribution P(X,Y,Z).
	 * @param condVar
	 *            Conditional variable Z.
	 * @return The conditional mutual information I(X;Y|Z).
	 */
	public static double computeConditionalMutualInformation(Function dist,
			DiscreteVariable condVar) {
		// ensure the distribution contains three variables
		assert dist.getDimension() == 3;

		// ensure the distribution contains the conditional variable
		assert dist.contains(condVar);

		// ensure the distribution sum up to one
		assert dist.sumUp() == 1.0;

		// I(X;Y|Z) = sum_X,Y,Z P(X,Y,Z) log P(X,Y|Z)/P(X|Z)P(Y|Z)
		// = sum_X,Y,Z P(X,Y,Z) log P(X,Y,Z)P(Z)/P(X,Z)P(Y,Z)
		List<DiscreteVariable> vars = dist.getVariables();
		DiscreteVariable x = vars.get(0);
		DiscreteVariable y = vars.get(1);
		if (x == condVar) {
			x = vars.get(2);
		}
		if (y == condVar) {
			y = vars.get(2);
		}

		Function pxz = dist.sumOut(y);
		Function pyz = dist.sumOut(x);
		Function pz = pxz.sumOut(x);

		// cells of joint, numerator, and denominator
		double[] distCells = dist._cells;
		double[] numCells = dist.times(pz)._cells;
		double[] denomCells = pxz.times(pyz)._cells;
		int size = dist.getDomainSize();

		double cmi = 0.0;
		double distCell;
		for (int i = 0; i < size; i++) {
			distCell = distCells[i];

			// if P(x,y,z) = 0, skip this term
			if (distCell != 0.0) {
				cmi += distCell * Math.log(numCells[i] / denomCells[i]);
			}
		}

		return cmi;
	}

	/**
	 * Returns the pairwise mutual information given the 2D distribution.
	 * 
	 * @param dist
	 *            distribution over a pair of variables.
	 * @return the pairwise mutual information given the 2D distribution.
	 */
	public static double computeMutualInformation(Function dist) {
		// ensure the distribution contains a pair of variables
		assert dist.getDimension() == 2;

		// ensure the distribution sum up to one
		assert dist.sumUp() == 1.0;

		// cells of joint and two marginal distributions
		double[] cells = dist._cells;
		double[] cells1 = dist.sumOut(dist.getVariables().get(1))._cells;
		double[] cells2 = dist.sumOut(dist.getVariables().get(0))._cells;

		// I(X;Y) = sum_X,Y P(X,Y) log P(X,Y)/P(X)P(Y)
		double mi = 0.0;
		int index = 0;
		for (double cell1 : cells1) {
			for (double cell2 : cells2) {
				double cell = cells[index++];

				// if P(x, y) = 0, skip this term
				if (cell != 0.0) {
					mi += cell * Math.log(cell / (cell1 * cell2));
				}
			}
		}

		return mi;
	}

	/**
	 * Returns the pairwise normalized mutual information given the 2D
	 * distribution.
	 * 
	 * @param dist
	 *            distribution over a pair of variables.
	 * @return the pairwise normalized mutual information given the 2D
	 *         distribution.
	 */
	public static double computeNormalizedMutualInformation(Function dist) {
		// ensure the distribution contains a pair of variables
		assert dist.getDimension() == 2;

		// ensure the distribution sum up to one
		assert dist.sumUp() == 1.0;

		// cells of joint and two marginal distributions
		double[] cells = dist._cells;
		double[] cells1 = dist.sumOut(dist.getVariables().get(1))._cells;
		double[] cells2 = dist.sumOut(dist.getVariables().get(0))._cells;

		// I(X;Y) = sum_X,Y P(X,Y) log P(X,Y)/P(X)P(Y)
		double mi = 0.0;
		int index = 0;
		for (double cell1 : cells1) {
			for (double cell2 : cells2) {
				double cell = cells[index++];

				// if P(x, y) = 0, skip this term
				if (cell != 0.0) {
					mi += cell * Math.log(cell / (cell1 * cell2));
				}
			}
		}

		// H(X) = -sum_X P(X) log P(X)
		double entropy1 = 0;
		for (double cell1 : cells1) {
			if (cell1 != 0)
				entropy1 -= cell1 * Math.log(cell1);
		}

		double entropy2 = 0;
		for (double cell2 : cells2) {
			if (cell2 != 0)
				entropy2 -= cell2 * Math.log(cell2);
		}

		// NMI(X;Y) = I(X;Y)/sqrt(H(X)H(Y))
		double nmi = mi == 0 ? 0 : mi / Math.sqrt(entropy1 * entropy2);

		return nmi;
	}

	/**
	 * Returns the asymmetric normalized mutual information, MI(X;Y)/H(X), given
	 * the 2D distribution P(X,Y).
	 * 
	 * @param dist
	 *            distribution over a pair of variables.
	 * @return asymmetric normalized mutual information
	 */
	public static double computeAsymmetricNMI(
			Function dist) {
		// ensure the distribution contains a pair of variables
		assert dist.getDimension() == 2;

		// ensure the distribution sum up to one
		assert dist.sumUp() == 1.0;

		// cells of joint and two marginal distributions
		double[] cells = dist._cells;
		double[] cells1 = dist.sumOut(dist.getVariables().get(1))._cells;
		double[] cells2 = dist.sumOut(dist.getVariables().get(0))._cells;

		// I(X;Y) = sum_X,Y P(X,Y) log P(X,Y)/P(X)P(Y)
		double mi = 0.0;
		int index = 0;
		for (double cell1 : cells1) {
			for (double cell2 : cells2) {
				double cell = cells[index++];

				// if P(x, y) = 0, skip this term
				if (cell != 0.0) {
					mi += cell * Math.log(cell / (cell1 * cell2));
				}
			}
		}

		// H(X) = -sum_X P(X) log P(X)
		double entropy1 = 0;
		for (double cell1 : cells1) {
			if (cell1 != 0)
				entropy1 -= cell1 * Math.log(cell1);
		}

		double anmi = mi == 0 ? 0 : mi / entropy1;

		return anmi;
	}

	/**
	 * Returns the entropy of the specified distribution.
	 * 
	 * @param dist
	 *            the distribution whose entropy is to be computed.
	 * @return the entropy of the specified distribution.
	 */
	public static double computeEntropy(Function dist) {
		// ensure the distribution sum up to one
		assert dist.sumUp() == 1.0;

		// H(X) = - sum_X P(X) log P(X)
		double ent = 0.0;
		for (double cell : dist._cells) {
			// if P(x) = 0, skip this term
			if (cell != 0.0) {
				ent -= cell * Math.log(cell);
			}
		}

		return ent;
	}

	public static double computeConditionalEntropy(Function dist,
			DiscreteVariable conditional) {
		double entropy = 0;

		double[] marginal = dist.marginalize(conditional).getCells();

		for (int state = 0; state < conditional.getCardinality(); state++) {
			Function conditionalProbability = dist.project(conditional, state);
			conditionalProbability.normalize();
			entropy += marginal[state] * computeEntropy(conditionalProbability);
		}

		return entropy;
	}

	/**
	 * Returns the mutual information I(v1;v2)
	 * 
	 * @param dist
	 * @param v1
	 * @param v2
	 * @return
	 */
	public static double computeMutualInformation(Function dist,
			Collection<DiscreteVariable> v1, DiscreteVariable v2) {
		// I(v1;v2) = H(v1) - H(v1|v2)
		return computeEntropy(dist.marginalize(v1))
				- computeConditionalEntropy(dist, v2);
	}

	/**
	 * Returns the normalized mutual information NMI({@code v1}; {@code v2})
	 * 
	 * @param dist
	 *            joint distribution of v1 and v2
	 * @param v1
	 *            first set of variables
	 * @param v2
	 *            second set of single variable
	 * @return the normalized mutual information
	 */
	public static double computeNormalizedMutualInformation(Function dist,
			Collection<DiscreteVariable> v1, DiscreteVariable v2) {
		double denominator =
				Math.sqrt(computeEntropy(dist.marginalize(v1))
						* computeEntropy(dist.marginalize(v2)));
		double mi = computeMutualInformation(dist, v1, v2);
		return mi == 0 ? 0 : mi / denominator;
	}

	/**
	 * Returns the KL divergence D(P||Q) between the distributions P and Q. The
	 * convention we use: (1) 0 * log(0) = 0; (2) x * log(0) = 0 (!!!)
	 * 
	 * @param p
	 *            distribution P.
	 * @param q
	 *            distribution Q.
	 * @return the KL divergence D(P||Q).
	 */
	public static double computeKl(Function p, Function q) {
		// ensure two functions over same domain
		assert Arrays.equals(p._variables, q._variables);

		double kl = 0.0;
		double[] pCells = p.getCells();
		double[] qCells = q.getCells();
		for (int i = 0; i < pCells.length; i++) {
			// skip cells where either P or Q instantiate to 0
			if (pCells[i] != 0.0 && qCells[i] != 0.0) {
				kl += pCells[i] * Math.log(pCells[i] / qCells[i]);
			}
		}

		return kl;
	}

}
