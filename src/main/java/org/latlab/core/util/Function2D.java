/**
 * Function2D.java 
 * Copyright (C) 2006 Tao Chen, Kin Man Poon, Yi Wang, and Nevin L. Zhang
 */
package org.latlab.core.util;

/**
 * This class provides an implementation for two-dimensional tabular functions,
 * namely, matrices.
 * 
 * @author Yi Wang
 * 
 */
class Function2D extends Function {

	/**
	 * the shortcut to the only two variables in this function. There is a
	 * requirement that _x>_y.
	 */
	protected DiscreteVariable _x, _y;

	/**
	 * <p>
	 * Constructs a function of the specified array of variables.
	 * </p>
	 * 
	 * <p>
	 * Note: Only function classes are supposed to call this method.
	 * </p>
	 * 
	 * @param variables
	 *            array of variables to be involved. There are two Variables
	 *            soorted in the Variable array.
	 */
	protected Function2D(DiscreteVariable[] variables) {
		super(variables);

		_x = _variables[0];
		_y = _variables[1];
	}

	/**
	 * <p>
	 * Constructs a function with all its internal data structures specified.
	 * </p>
	 * 
	 * <p>
	 * Note: Only function classes are supposed to call this method.
	 * </p>
	 * 
	 * @param variables
	 *            array of variables in new function. There are two Variables in
	 *            the Variable array.
	 * @param cells
	 *            array of cells in new function.
	 * @param magnitudes
	 *            array of magnitudes for variables in new function.
	 */
	protected Function2D(DiscreteVariable[] variables, double[] cells, int[] magnitudes) {
		super(variables, cells, magnitudes);

		_x = _variables[0];
		_y = _variables[1];
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.latlab.core.util.Function#normalize(org.latlab.core.util.Variable)
	 */
	public final boolean normalize(DiscreteVariable variable) {

		// For Test
		//System.out.println("Function2D.normalize(Variable) executed");

		// argument variable must be either of the variables in this function
		assert variable == _x || variable == _y;

		boolean hasZero = false;

		int xCard = _x.getCardinality();
		int yCard = _y.getCardinality();

		int index;
		double sum;

		if (variable == _x) {
			// uniform probability that may be used
			double uniform = 1.0 / xCard;

			for (int i = 0; i < yCard; i++) {
				// computes sum
				index = i;
				sum = 0.0;
				for (int j = 0; j < xCard; j++) {
					sum += _cells[index];
					index += yCard;
				}

				// normalizes
				index = i;
				if (sum != 0.0) {
					for (int j = 0; j < xCard; j++) {
						_cells[index] /= sum;
						index += yCard;
					}
				} else {
					for (int j = 0; j < xCard; j++) {
						_cells[index] = uniform;
						index += yCard;
					}

					hasZero = true;
				}
			}
		} else {
			// uniform probability that may be used
			double uniform = 1.0 / yCard;

			index = 0;
			for (int i = 0; i < xCard; i++) {
				// computes sum
				sum = 0.0;
				for (int j = 0; j < yCard; j++) {
					sum += _cells[index++];
				}

				// normalizes
				index -= yCard;
				if (sum != 0.0) {
					for (int j = 0; j < yCard; j++) {
						_cells[index++] /= sum;
					}
				} else {
					for (int j = 0; j < yCard; j++) {
						_cells[index++] = uniform;
					}

					hasZero = true;
				}
			}
		}

		return hasZero;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.latlab.core.util.Function#project(org.latlab.core.util.Variable, int)
	 */
	public Function project(DiscreteVariable variable, int state) {

		// For Test
		//System.out.println("Function2D.project(Variable, int) executed");

		// argument variable must be either of the variables in this function
		assert variable == _x || variable == _y;

		// state must be valid
		assert variable.isValid(state);

		// result is an one-dimensional function
		DiscreteVariable[] variables;
		double[] cells;
		int[] magnitudes = new int[] { 1 };

		if (variable == _x) {
			variables = new DiscreteVariable[] { _y };

			int yCard = _y.getCardinality();
			cells = new double[yCard];

			System.arraycopy(_cells, state * yCard, cells, 0, yCard);
		} else {
			variables = new DiscreteVariable[] { _x };

			int xCard = _x.getCardinality();
			int yCard = _y.getCardinality();
			cells = new double[xCard];

			int index = state;
			for (int i = 0; i < xCard; i++) {
				cells[i] = _cells[index];
				index += yCard;
			}
		}

		return (new Function1D(variables, cells, magnitudes));
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.latlab.core.util.Function#sumOut(org.latlab.core.util.Variable)
	 */
	public Function sumOut(DiscreteVariable variable) {

		// For Test
		//System.out.println("Function2D.sumOut(Variable) executed");

		// argument variable must be either of the variables in this function
		assert variable == _x || variable == _y;

		// result is an one-dimensional function
		DiscreteVariable[] variables;
		double[] cells;
		int[] magnitudes = new int[] { 1 };

		int xCard = _x.getCardinality();
		int yCard = _y.getCardinality();

		if (variable == _x) {
			variables = new DiscreteVariable[] { _y };

			cells = new double[yCard];

			int index = 0;
			for (int i = 0; i < xCard; i++) {
				for (int j = 0; j < yCard; j++) {
					cells[j] += _cells[index++];
				}
			}
		} else {
			variables = new DiscreteVariable[] { _x };

			cells = new double[xCard];

			int index = 0;
			for (int i = 0; i < xCard; i++) {
				for (int j = 0; j < yCard; j++) {
					cells[i] += _cells[index++];
				}
			}
		}

		return (new Function1D(variables, cells, magnitudes));
	}

	/**
	 * Returns the product between this Function2D and another function. The
	 * multiplication is delegated to <code>Function1D.times(Function)</code>
	 * if the argument is a Function1D and they share a common Variable.
	 * 
	 * @param function
	 *            another factor
	 * @return the product between this Function1D and another function.
	 * @see Function1D#times(Function)
	 */
	public final Function times(Function function) {
		if (function instanceof Function1D
				&& this.contains(function._variables[0])) {
			return ((Function1D) function).times(this);
		} else if (function instanceof Function1D
				&& _x == function._variables[0] && _y == function._variables[1]) {
			Function result = this.clone();
			for (int i = 0; i < getDomainSize(); i++) {
				result._cells[i] *= function._cells[i];
			}
			//System.out.println("Function2DxFunction2D called");
			return result;
		} else {
			return super.times(function);
		}
	}

	public static void main(String[] args) {

		int[] a = new int[] { 1, 2, 3 };
		int index = 0;
		a[index] = a[index++];
		System.out.println(index + " " + a[0] + a[1] + a[2]);

	}

}
