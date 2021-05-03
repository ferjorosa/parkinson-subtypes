package org.latlab.core.model;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.math.Functions;
import org.latlab.core.util.*;

import java.util.*;

/**
 * Potential used in the representation of a conditional Gaussian Bayesian
 * network.
 * 
 * <p>
 * It holds a joint continuous variable as head variable and a discrete variable
 * as tail variable. If any of these variables are changed, this potential may
 * need to be updated to maintain in a valid state.
 * 
 * @author leonard
 * 
 */
public class CGPotential implements Potential {
	private List<SingularContinuousVariable> continuousVariables;
	private DiscreteVariable discreteVariable;

	private CGParameter[] parameters;

	/**
	 * Constructs a CG potential from a head joint variable and a parent
	 * discrete variable.
	 * 
	 * <p>
	 * It currently supports only one discrete variable.
	 * 
	 * @param jointVariable
	 *            head joint variable, which contains more than one singular
	 *            variable
	 * @param discreteVariable
	 *            discrete parent variable, with {@code null} to indicate no
	 *            discrete parent
	 */
	public CGPotential(JointContinuousVariable jointVariable,
			DiscreteVariable discreteVariable) {
		this(jointVariable.variables(), discreteVariable, true);
	}

	/**
	 * Constructs a CG potential from a head joint variable and a parent
	 * discrete variable. The given {@code parameters} will be used as the
	 * parameters of this potential.
	 * 
	 * <p>
	 * It currently supports only one discrete variable.
	 * 
	 * @param jointVariable
	 *            head joint variable, which contains more than one singular
	 *            variable
	 * @param discreteVariable
	 *            discrete parent variable, with {@code null} to indicate no
	 *            discrete parent
	 * @param parameters
	 *            used as the parameters of the new potential
	 */
	public CGPotential(JointContinuousVariable jointVariable,
			DiscreteVariable discreteVariable, CGParameter[] parameters) {
		this(jointVariable.variables(), discreteVariable, false);

		assert this.parameters.length == parameters.length;
		System.arraycopy(parameters, 0, this.parameters, 0, parameters.length);
	}

	/**
	 * Constructs a CG potential from a head joint variable and a parent
	 * discrete variable.
	 * 
	 * <p>
	 * It currently supports only one discrete variable.
	 * 
	 * @param jointVariable
	 *            head joint variable, which contains more than one singular
	 *            variable
	 * @param discreteVariable
	 *            discrete parent variable, with {@code null} to indicate no
	 *            discrete parent
	 * @param allocate
	 *            whether to allocate memory for each parameter
	 */
	private CGPotential(
			Collection<SingularContinuousVariable> continuousVariables,
			DiscreteVariable discreteVariable, boolean allocate) {
		assert continuousVariables.size() > 0;

		this.continuousVariables =
				new ArrayList<SingularContinuousVariable>(continuousVariables);
		this.discreteVariable = discreteVariable;

		resetParameters(allocate);
	}

	/**
	 * Resets the parameters according to the list of discrete variables and
	 * continuous variables. The old parameters will be discarded.
	 * 
	 * @param allocate
	 *            whether to allocate the memory for each parameter
	 */
	private void resetParameters(boolean allocate) {
		int length =
				discreteVariable == null ? 1
						: discreteVariable.getCardinality();
		parameters = new CGParameter[length];

		if (allocate) {
			for (int i = 0; i < parameters.length; i++) {
				parameters[i] = new CGParameter(continuousVariables.size());
			}
		}
	}

	/**
	 * Returns the parameter at the index.
	 * 
	 * @param index
	 *            index of the parameter
	 * @return parameter at the index
	 */
	public CGParameter get(int index) {
		return parameters[index];
	}

    /**
     * Sets the parameter at the index
     * @param index
     * @param parameter
     */
	public void set(int index, CGParameter parameter) {
		this.parameters[index] = parameter;
	}

	/**
	 * @see Potential#addParentVariable(Variable)
	 */
	public CGPotential addParentVariable(Variable variable) {
		variable.accept(new Variable.Visitor<Void>() {
			@Override
			public Void visit(SingularContinuousVariable variable) {
				throw new UnsupportedOperationException(
						"A continuous variable is not expected to have more than one continuous head variables or any tail continuous variable yet.");
			}

			@Override
			public Void visit(JointContinuousVariable variable) {
				throw new UnsupportedOperationException(
						"A continuous variable is not expected to have more than one continuous head variables or any tail continuous variable yet.");
			}

			@Override
			public Void visit(DiscreteVariable variable) {
				if (discreteVariable != null)
					throw new UnsupportedOperationException(
							"A continuous variable is not expected to have more than one discrete parent variable yet.");

				discreteVariable = variable;
				resetParameters(true);
				return null;
			}
		});

		return this;
	}

	/**
	 * @see Potential#removeParentVariable(Variable)
	 */
	public CGPotential removeParentVariable(Variable variable) {
		assert discreteVariable == variable;
		discreteVariable = null;
		resetParameters(true);
		return this;
	}

	public void addHeadVariable(Collection<SingularContinuousVariable> variables) {
		if (continuousVariables.addAll(variables))
			resetParameters(true);
	}

	public void removeHeadVariable(
			Collection<SingularContinuousVariable> variables) {
		if (continuousVariables.removeAll(variables))
			resetParameters(true);
	}

	/**
	 * Returns the expected number of entries for setting the values of this
	 * potential.
	 * 
	 * @return expected number of entries
	 */
	public int expectedNumberOfEntries() {
		int entryPerConfig = expectedNumberOfEntriesPerConfig();

		int numberOfConfig =
				discreteVariable == null ? 1
						: discreteVariable.getCardinality();
		return entryPerConfig * numberOfConfig;
	}

	/**
	 * Returns the expected number of entries for setting the values of this
	 * potential for each configuration of discrete parent variables.
	 * 
	 * @return the expected number of entries for setting the values of this
	 *         potential
	 */
	public int expectedNumberOfEntriesPerConfig() {
		int dimension = continuousVariables.size();
		return (dimension + 1) * dimension;
	}

	/**
	 * Returns the number of parameters in this potential. This is the product
	 * of the cardinalities of the discrete parent variables.
	 * 
	 * @return number of parameters
	 */
	public int size() {
		return parameters.length;
	}

	/**
	 * Sets the values of this potential according to the given {@code entries}.
	 * The entries are in the order of the configuration of the parents, and for
	 * each configuration, they are listed by entries in the mean vector and
	 * covariance matrix. The matrix are listed row by row.
	 * 
	 * <p>
	 * Current it supports only one parent variable.
	 * 
	 * @param parents
	 *            parent variables in the order of the configuration
	 * @param entries
	 *            entries for setting values
	 */
	public void setEntries(List<DiscreteVariable> parents, List<Double> entries) {
		int block = expectedNumberOfEntriesPerConfig();
		int current = 0;
		int index = 0;
		while (current < entries.size()) {
			setEntries(parents, Arrays.asList(index), entries.subList(current,
					current + block));
			current += block;
			index++;
		}
	}

	public List<Double> getEntries(List<DiscreteVariable> parents) {
		List<Double> entries = new ArrayList<Double>(expectedNumberOfEntries());
		DiscreteVariable parent = parents.get(0);
		for (int i = 0; i < parent.getCardinality(); i++) {
			entries.addAll(get(i).getEntries());
		}

		return entries;
	}

	/**
	 * Sets the values of this potential for one particular configuration.
	 * 
	 * @see setEntries(List<DiscreteVariable>, List<Double>)
	 * @param parents
	 *            parent variables
	 * @param states
	 *            configuration
	 * @param entries
	 *            entries for setting values
	 */
	public void setEntries(List<DiscreteVariable> parents,
			List<Integer> states, List<Double> entries) {
		parameters[states.get(0)].setEntries(entries);
	}

	public DiscreteVariable discreteVariable() {
		return discreteVariable;
	}

	public List<SingularContinuousVariable> continuousVariables() {
		return Collections.unmodifiableList(continuousVariables);
	}

	public CGPotential clone() {
		CGPotential result =
				new CGPotential(continuousVariables, discreteVariable, false);
		for (int i = 0; i < parameters.length; i++) {
			result.parameters[i] = parameters[i].copy();
		}

		return result;
	}

	/**
	 * Returns the natural log of the constant multiplied to {@p} value of each
	 * parameter.
	 * 
	 * @param variable
	 * @param value
	 * @return
	 */
	public double absorbEvidence(SingularContinuousVariable variable,
			double value) {
		int evidenceIndex = continuousVariables.indexOf(variable);
		int[] variableIndices = createVariableIndicesWithout(evidenceIndex);

		double[] logP = new double[parameters.length];

		// uses the maximum for normalization, so that the smallest p value
		// becomes zero. If we use minimum, the largest p value may become
		// infinity.
		double maxLogP = Double.NEGATIVE_INFINITY;

		for (int i = 0; i < parameters.length; i++) {
			CGParameter parameter = parameters[i];

			// scalars derived from old parameter
			double A_E = parameter.A.getQuick(evidenceIndex);
			double C_EE = parameter.C.getQuick(evidenceIndex, evidenceIndex);

			// vectors derived from old parameter
			DoubleMatrix1D A_Y = parameter.A.viewSelection(variableIndices);
			DoubleMatrix1D C_YE =
					parameter.C.viewColumn(evidenceIndex).viewSelection(
							variableIndices);
			DoubleMatrix1D C_EY =
					parameter.C.viewRow(evidenceIndex).viewSelection(
							variableIndices);

			// matrix derived from old parameter
			DoubleMatrix2D C_YY =
					parameter.C.viewSelection(variableIndices, variableIndices);

			// calculate the new p
			// p = p * N(A_E, C_EE)|e, which is the pdf of the normal function
			// at e.
			// parameter.p *= Normal.pdf(A_E, C_EE, value);

			logP[i] = Normal.logPdf(A_E, C_EE, value);
			if (logP[i] > maxLogP)
				maxLogP = logP[i];

			// calculate the new A
			// A_Y = A_Y + C_YE * (e - A_E) / C_EE, A_E = E
			A_Y.assign(C_YE, Functions.plusMult((value - A_E) / C_EE));
			parameter.A.setQuick(evidenceIndex, value);

			// calculate the new C
			// C_YY = C_YY - C_YE C_EY / C_EE, other entries become 0
			DoubleMatrix2D product =
					Algebra.DEFAULT.multOuter(C_YE, C_EY, null);
			C_YY.assign(product, Functions.minusMult(1 / C_EE));
			C_YE.assign(0);
			C_EY.assign(0);
			parameter.C.setQuick(evidenceIndex, evidenceIndex, 0);
		}

		for (int i = 0; i < parameters.length; i++) {
			parameters[i].p *= Math.exp(logP[i] - maxLogP);
		}

		return maxLogP;
	}

	public void timesIndicator(DiscreteVariable variable, int state) {
		if (discreteVariable != variable) {
			throw new IllegalArgumentException(
					"This clique does not contain the requested variable.");
		}

		for (int i = 0; i < parameters.length; i++) {
			if (i != state) {
				parameters[i].p = 0;
			}
		}
	}

	/**
	 * Creates an array of the indices of the variables, without the given
	 * {@code index}.
	 * 
	 * @param index
	 * @return
	 */
	private int[] createVariableIndicesWithout(int index) {
		int result[] = new int[continuousVariables.size() - 1];
		int current = 0;

		for (int i = 0; i < result.length; i++) {
			if (current == index)
				current++;

			result[i] = current;
			current++;
		}

		return result;
	}

	public double mean() {
		assert continuousVariables.size() == 1;

		double sum = 0;
		for (CGParameter parameter : parameters) {
			sum += parameter.p * parameter.A.getQuick(0);
		}

		return sum;
	}

	public double variance() {
		assert continuousVariables.size() == 1;

		double mean = mean();

		double sum = 0;
		for (CGParameter parameter : parameters) {
			double difference = parameter.A.getQuick(0) - mean;
			sum +=
					parameter.p
							* (parameter.C.getQuick(0, 0) + difference
									* difference);
		}

		return sum;
	}

	public Function marginalize(DiscreteVariable variable) {
		assert variable == discreteVariable;
		return function();
	}

	public void multiply(Function p) {
		double[] cells = p.getCells();
		assert cells.length == parameters.length;

		for (int i = 0; i < parameters.length; i++) {
			parameters[i].p *= cells[i];
		}
	}

	/**
	 * Copies from the vector A and matrix C of another potential.
	 * 
	 * @param other
	 *            potential to copy from
	 * 
	 */
	public void combine(CGPotential other) {
		assert parameters.length == other.parameters.length;

		for (int i = 0; i < parameters.length; i++) {
			CGParameter p1 = parameters[i];
			CGParameter p2 = other.parameters[i];

			p1.p *= p2.p;
			p1.A.assign(p2.A);
			p1.C.assign(p2.C);
		}
	}

	public double normalize(double constant) {
		if (Double.isNaN(constant)) {
			constant = 0;
			for (CGParameter parameter : parameters) {
				constant += parameter.p;
			}
		}

		if (constant == 0) {
			for (CGParameter parameter : parameters) {
				parameter.p = 1d / parameters.length;
			}
		} else {
			for (CGParameter parameter : parameters) {
				parameter.p /= constant;
			}
		}

		return constant;
	}

	/**
	 * Marginalizes this potential to the given continuous variable. The
	 * discrete parents are kept the same.
	 * 
	 * @param variable
	 * @return
	 */
	public CGPotential marginalize(SingularContinuousVariable variable) {
		int index = continuousVariables.indexOf(variable);
		if (index < 0)
			throw new IllegalArgumentException();

		CGPotential marginal =
				new CGPotential(Arrays.asList(variable), discreteVariable, true);
		for (int i = 0; i < parameters.length; i++) {
			CGParameter mp = marginal.parameters[i];
			CGParameter cp = parameters[i];
			mp.p = cp.p;
			mp.A.setQuick(0, cp.A.getQuick(index));
			mp.C.setQuick(0, 0, cp.C.getQuick(index, index));
		}

		return marginal;
	}

	public Function function() {
		Function function =
				Function.createFunction(Arrays.asList(discreteVariable));
		double[] cells = function.getCells();
		for (int i = 0; i < parameters.length; i++) {
			cells[i] = parameters[i].p;
		}

		return function;
	}

	public void generateRandomParameters() {
		int length = continuousVariables.size();
		for (CGParameter parameter : parameters) {
			for (int i = 0; i < length; i++) {
				parameter.C.setQuick(i, i, 1);
			}
		}
	}

	@Override
	public String toString() {
		return getClass().getSimpleName() + "\n" + Arrays.toString(parameters);
	}

	public void reorderStates(DiscreteVariable variable, int[] order) {
		assert order.length == parameters.length;
		CGParameter[] newParameters = new CGParameter[parameters.length];

		for (int i = 0; i < order.length; i++) {
			newParameters[i] = parameters[order[i]];
		}

		parameters = newParameters;
	}
}
