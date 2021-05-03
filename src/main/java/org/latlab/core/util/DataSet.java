/**
 * DataSet.java 
 * Copyright (C) 2006 Tao Chen, Kin Man Poon, Yi Wang, and Nevin L. Zhang
 */
package org.latlab.core.util;

import cern.jet.random.Uniform;
import org.latlab.core.model.BayesNet;

import java.io.*;
import java.util.*;

/**
 * This class provides an implementation for data sets.
 * 
 * @author Yi Wang
 * 
 */
public final class DataSet {

	/**
	 * This class provides an implementation for data cases.
	 * 
	 * @author Yi Wang
	 * 
	 */
	public final class DataCase implements Comparable<DataCase> {
		/**
		 * the data set to which this data case belongs.
		 */
		private DataSet _dataSet;

		/**
		 * the array of states of this data case.
		 */
		private int[] _states;

		/**
		 * the weight of this data case.
		 */
		private double _weight;

		/**
		 * Constructs a data case with the specified data set that contains it,
		 * and the specified states and weight.
		 * 
		 * @param dataSet
		 *            data set that contains this data case.
		 * @param states
		 *            states of this data case.
		 * @param weight
		 *            weight of this data case.
		 */
		private DataCase(DataSet dataSet, int[] states, double weight) {
			_dataSet = dataSet;
			_states = states;
			_weight = weight;
		}

		/**
		 * <p>
		 * Compares this data case with the specified object for order.
		 * </p>
		 * 
		 * <p>
		 * If the specified object is not a data case, this method will throw a
		 * <code>ClassCastException</code> (as data cases are comparable only
		 * to other data cases). Otherwise, the comparison is carried out based
		 * on the states of two data cases.
		 * </p>
		 * 
		 * @param object
		 *            the object to be compared.
		 * @return a negative or a positive integer if the states of this data
		 *         case procedes or succeeds that of the specified data case;
		 *         zero if their states are identical.
		 */
		public int compareTo(DataCase object) {
			DataCase dataCase = object;

			// two data cases must belong to the same data set
			assert _dataSet == dataCase._dataSet;

			for (int i = 0; i < _states.length; i++) {
				if (_states[i] < dataCase._states[i]) {
					return -1;
				} else if (_states[i] > dataCase._states[i]) {
					return 1;
				}
			}

			return 0;
		}

		/**
		 * Returns the states of this data case.
		 * 
		 * @return the states of this data case
		 */
		public int[] getStates() {
			return _states;
		}

		/**
		 * Returns the weight of this data case.
		 * 
		 * @return the weight of this data case.
		 */
		public double getWeight() {
			return _weight;
		}

		/**
		 * Updates the weight of this data case.
		 * 
		 * @param weight
		 *            new weight of this data case.
		 */
		public void setWeight(double weight) {
			// weight must be positive
			assert weight > 0.0;

			_weight = weight;
		}
	}

	/**
	 * the constant for missing value.
	 */
	public final static int MISSING_VALUE = -1;

	/**
	 * the prefix of default names of data sets.
	 */
	private final static String NAME_PREFIX = "DataSet";

	/**
	 * the number of created data sets.
	 */
	private static int _count = 0;

	/**
	 * the name of this data set.
	 */
	protected String _name;

	/**
	 * the array of variables involved in this data set.
	 */
	private DiscreteVariable[] _variables;

	/**
	 * the list of distinct data cases. we use <code>ArrayList</code> for
	 * random access.
	 */
	private ArrayList<DataCase> _data;

	/**
	 * the total weight, namely, number of data cases, of this data set.
	 */
	private double _totalWeight;

	/**
	 * the flag that indicates whether this data set contains missing values.
	 */
	private boolean _missing;

	/**
	 * Reads the data set from the specified input stream.
	 * 
	 * @param stream
	 *            input stream of the data set
	 * @throws IOException
	 *             thrown when an IO error occurs
	 */
	public DataSet(InputStream stream) throws IOException {
		StreamTokenizer tokenizer = new StreamTokenizer(new BufferedReader(
				new InputStreamReader(stream, "UTF8")));

		tokenizer.resetSyntax();

		// characters that will be ignored
		tokenizer.whitespaceChars(':', ':');
		tokenizer.whitespaceChars(' ', ' ');
		tokenizer.whitespaceChars('\t', '\t');

		// word characters
		tokenizer.wordChars('A', 'z');
		tokenizer.wordChars('\'', '\'');
		tokenizer.wordChars('(', '(');
		tokenizer.wordChars(')', ')');
		tokenizer.wordChars(']', ']');
		tokenizer.wordChars('[', '[');
		tokenizer.wordChars('-', '-');
		tokenizer.wordChars('0', '9');

		// we will parse numbers
		tokenizer.parseNumbers();

		// treats eol as a token
		tokenizer.eolIsSignificant(true);

		// ignores c++ comments
		tokenizer.slashSlashComments(true);

		// starts parsing
		int value = StreamTokenizer.TT_EOF;

		// name of data set
		do {
			value = tokenizer.nextToken();
		} while (value != StreamTokenizer.TT_WORD);

		do {
			value = tokenizer.nextToken();
		} while (value != StreamTokenizer.TT_WORD);
		_name = tokenizer.sval;

		// reads variables
		ArrayList<DiscreteVariable> variables = new ArrayList<DiscreteVariable>();

		while (true) {
			// goes to the next word or number
			do {
				value = tokenizer.nextToken();
			} while (!(value == StreamTokenizer.TT_WORD || value == StreamTokenizer.TT_NUMBER));

			// if we see a number, we have read all variables
			if (value == StreamTokenizer.TT_NUMBER) {
				tokenizer.pushBack();
				break;
			}

			// we have at least one more variable to read
			String name = tokenizer.sval;

			// read state names
			ArrayList<String> states = new ArrayList<String>();
			do {
				value = tokenizer.nextToken();
				if (value == StreamTokenizer.TT_WORD) {
					states.add(tokenizer.sval);
				}
			} while (value != StreamTokenizer.TT_EOL);

			variables.add(new DiscreteVariable(name, states));
		}

		// enforces order of variables
		_variables = variables.toArray(new DiscreteVariable[variables.size()]);
		Arrays.sort(_variables);

		// reads data cases
		_data = new ArrayList<DataCase>();
		while (true) {
			// goes to the next number
			do {
				value = tokenizer.nextToken();
			} while (value != StreamTokenizer.TT_EOF
					&& value != StreamTokenizer.TT_NUMBER);

			// we have at least one more data case
			if (value == StreamTokenizer.TT_NUMBER) {
				tokenizer.pushBack();
			} else {
				break;
			}

			// reads states
			int[] states = new int[getDimension()];
			for (int i = 0; i < getDimension(); i++) {
				do {
					value = tokenizer.nextToken();
				} while (value != StreamTokenizer.TT_NUMBER);
				states[i] = (int) tokenizer.nval;
			}

			// reads weight
			do {
				value = tokenizer.nextToken();
			} while (value != StreamTokenizer.TT_NUMBER);
			double weight = tokenizer.nval;

			// adds data case
			addDataCase(states, weight);

			do {
				value = tokenizer.nextToken();
			} while (value != StreamTokenizer.TT_EOL
					&& value != StreamTokenizer.TT_EOF);

			if (value == StreamTokenizer.TT_EOF) {
				break;
			}
		}

		_count++;
	}

	/**
	 * Constructs a data set defined by the specified data file.
	 * 
	 * @param file
	 *            The name of the specified data file.
	 * @throws IOException
	 *             if an I/O error occurs.
	 */
	public DataSet(String file) throws IOException {
		StreamTokenizer tokenizer = new StreamTokenizer(new InputStreamReader(
				new FileInputStream(file), "UTF8"));

		tokenizer.resetSyntax();

		// characters that will be ignored
		tokenizer.whitespaceChars(':', ':');
		tokenizer.whitespaceChars(' ', ' ');
		tokenizer.whitespaceChars('\t', '\t');

		// word characters
		tokenizer.wordChars('A', 'z');
		tokenizer.wordChars('\'', '\'');
		tokenizer.wordChars('(', '(');
		tokenizer.wordChars(')', ')');
		tokenizer.wordChars(']', ']');
		tokenizer.wordChars('[', '[');
		tokenizer.wordChars('-', '-');
		tokenizer.wordChars('0', '9');

		// we will parse numbers
		tokenizer.parseNumbers();

		// treats eol as a token
		tokenizer.eolIsSignificant(true);

		// ignores c++ comments
		tokenizer.slashSlashComments(true);

		// starts parsing
		int value = StreamTokenizer.TT_EOF;

		// name of data set
		do {
			value = tokenizer.nextToken();
		} while (value != StreamTokenizer.TT_WORD);

		do {
			value = tokenizer.nextToken();
		} while (value != StreamTokenizer.TT_WORD);
		_name = tokenizer.sval;

		// reads variables
		ArrayList<DiscreteVariable> variables = new ArrayList<DiscreteVariable>();

		while (true) {
			// goes to the next word or number
			do {
				value = tokenizer.nextToken();
			} while (!(value == StreamTokenizer.TT_WORD || value == StreamTokenizer.TT_NUMBER));

			// if we see a number, we have read all variables
			if (value == StreamTokenizer.TT_NUMBER) {
				tokenizer.pushBack();
				break;
			}

			// we have at least one more variable to read
			String name = tokenizer.sval;

			// read state names
			ArrayList<String> states = new ArrayList<String>();
			do {
				value = tokenizer.nextToken();
				if (value == StreamTokenizer.TT_WORD) {
					states.add(tokenizer.sval);
				}
			} while (value != StreamTokenizer.TT_EOL);

			variables.add(new DiscreteVariable(name, states));
		}

		// enforces order of variables
		_variables = variables.toArray(new DiscreteVariable[variables.size()]);
		Arrays.sort(_variables);

		// reads data cases
		_data = new ArrayList<DataCase>();
		while (true) {
			// goes to the next number
			do {
				value = tokenizer.nextToken();
			} while (value != StreamTokenizer.TT_EOF
					&& value != StreamTokenizer.TT_NUMBER);

			// we have at least one more data case
			if (value == StreamTokenizer.TT_NUMBER) {
				tokenizer.pushBack();
			} else {
				break;
			}

			// reads states
			int[] states = new int[getDimension()];
			for (int i = 0; i < getDimension(); i++) {
				do {
					value = tokenizer.nextToken();
				} while (value != StreamTokenizer.TT_NUMBER);
				states[i] = (int) tokenizer.nval;
			}

			// reads weight
			do {
				value = tokenizer.nextToken();
			} while (value != StreamTokenizer.TT_NUMBER);
			double weight = tokenizer.nval;

			// adds data case
			addDataCase(states, weight);

			do {
				value = tokenizer.nextToken();
			} while (value != StreamTokenizer.TT_EOL
					&& value != StreamTokenizer.TT_EOF);

			if (value == StreamTokenizer.TT_EOF) {
				break;
			}
		}

		_count++;
	}

	/**
	 * Constructs an empty data set of the specified array of variables.
	 * 
	 * @param variables
	 *            array of variables to be involved.
	 */
	public DataSet(DiscreteVariable[] variables) {
		// default name
		_name = NAME_PREFIX + _count++;

		// Added by Chen Tao
		_variables = variables;

		// enforces order of variables
		Arrays.sort(_variables);

		_data = new ArrayList<DataCase>();
	}

	/**
	 * Adds the specified data case with the specified weight to this data set.
	 * 
	 * @param states
	 *            data case to be added to this data set.
	 * @param weight
	 *            weight of the data case.
	 */
	public void addDataCase(int[] states, double weight) {
		DataCase dataCase = new DataCase(this, states, weight);

		// finds the position for this data case
		int index = Collections.binarySearch(_data, dataCase);

		if (index < 0) {
			// adds unseen data case
			_data.add(-index - 1, dataCase);

			// updates missing value flag
			for (int state : states) {
				_missing |= (state == MISSING_VALUE);
			}
		} else {
			// increases weight for existing data case
			dataCase = _data.get(index);
			dataCase.setWeight(dataCase.getWeight() + weight);
		}

		// updates total weight
		_totalWeight += weight;
	}

	/**
	 * Returns the list of distinct data cases in this data set.
	 * 
	 * @return the list of distinct data cases in this data set.
	 */
	public ArrayList<DataCase> getData() {
		return _data;
	}

	/**
	 * Returns the dimension, namely, the number of variables, of this data set.
	 * 
	 * @return the dimension of this data set.
	 */
	public int getDimension() {
		return _variables.length;
	}

	/**
	 * Returns the name of this data set.
	 * 
	 * @return the name of this data set.
	 */
	public String getName() {
		return _name;
	}

	/**
	 * Returns the number of distinct data cases in this data set.
	 * 
	 * @return the number of distinct data cases in this data set.
	 */
	public int getNumberOfEntries() {
		return _data.size();
	}

	/**
	 * Returns the total weight, namely, the number of data cases, of this data
	 * set.
	 * 
	 * @return the total weight of this data set.
	 */
	public double getTotalWeight() {
		return _totalWeight;
	}

	/**
	 * Returns the array of variables involved in this data set.
	 * 
	 * @return the array of variables involved in this data set.
	 */
	public DiscreteVariable[] getVariables() {
		return _variables;
	}

	/**
	 * Returns <code>true</code> if this data set contains missing values.
	 * 
	 * @return <code>true</code> if this data set contains missing values.
	 */
	public boolean hasMissingValues() {
		return _missing;
	}

	/**
	 * Returns a data set that is the projection of this data set on the
	 * specified list of variables.
	 * 
	 * @param variables
	 *            list of variables onto which this data set is to be project.
	 * @return a projected data set on the specified list of variables.
	 */
	public DataSet project(ArrayList<DiscreteVariable> variables) {
		// array representation
		DiscreteVariable[] varArray = variables.toArray(new DiscreteVariable[variables.size()]);

		// order of variables will be enforced in the constructor
		DataSet dataSet = new DataSet(varArray);

		// maps argument variables to that in this data set
		int dimension = variables.size();
		int[] map = new int[dimension];
		for (int i = 0; i < dimension; i++) {
			map[i] = Arrays.binarySearch(_variables, varArray[i]);

			// argument variable must be involved in this data set
			assert map[i] >= 0;
		}

		// projection
		// int[] projectedStates = new int[dimension];
		for (DataCase dataCase : _data) {
			int[] projectedStates = new int[dimension];

			int[] states = dataCase.getStates();
			for (int i = 0; i < dimension; i++) {
				projectedStates[i] = states[map[i]];
			}

			dataSet.addDataCase(projectedStates, dataCase.getWeight());
		}

		return dataSet;
	}

	/**
	 * Outputs this data set to the specified file.
	 * 
	 * @param file
	 *            output of this BN.
	 * @throws FileNotFoundException
	 *             if the file exists but is a directory rather than a regular
	 *             file, does not exist but cannot be created, or cannot be
	 *             opened for any other reason.
	 * @throws UnsupportedEncodingException
	 */
	public void save(String file) throws FileNotFoundException,
			UnsupportedEncodingException {
		PrintWriter out = new PrintWriter(new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(file), "UTF8")));

		// outputs header
		out.println("// " + file);
		out.println("// Produced by org.latlab at "
				+ (new Date(System.currentTimeMillis())));

		// outputs name
		out.println("Name: " + _name);
		out.println();

		// outputs variables
		out.println("// " + getDimension() + " variables");
		for (DiscreteVariable variable : _variables) {
			// variable name
			out.printf(variable.getName() + ":");

			// state names
			for (String state : variable.getStates()) {
				out.print(" " + state);
			}

			out.println();
		}
		out.println();

		// outputs data cases
		out.println("// " + getNumberOfEntries()
				+ " distinct data cases with total weight " + getTotalWeight());
		for (DataCase dataCase : _data) {
			for (int state : dataCase.getStates()) {
				out.print(state + " ");
			}

			// outputs weight
			out.println(dataCase.getWeight());
		}

		out.close();
	}

	/**
	 * Outputs this data set to the specified file in ARFF format.
	 * 
	 * @param file
	 *            output of this BN.
	 * @throws FileNotFoundException
	 *             if the file exists but is a directory rather than a regular
	 *             file, does not exist but cannot be created, or cannot be
	 *             opened for any other reason.
	 * @throws UnsupportedEncodingException
	 */
	public void saveAsArff(String file) throws FileNotFoundException,
			UnsupportedEncodingException {
		PrintWriter out = new PrintWriter(new BufferedWriter(
				new OutputStreamWriter(new FileOutputStream(file), "UTF8")));

		// output relation
		out.println("@relation " + _name);
		out.println();

		// output attributes
		int dim = getDimension();
		out.println("% " + dim + " variables");

		for (DiscreteVariable variable : _variables) {
			// output name
			out.print("@attribute " + variable.getName());

			// output domain
			out.print(" {");
			for (int k = 0; k < variable.getCardinality(); k++) {
				out.print(variable.getStates().get(k));
				if (k < variable.getCardinality() - 1) {
					out.print(",");
				}
			}
			out.println("}");
		}

		out.println();

		// output data cases
		out.println("% " + getNumberOfEntries() + " distinct records, "
				+ getTotalWeight() + " in total");
		out.println("@data");

		for (DataCase datum : _data) {
			StringBuffer states = new StringBuffer();

			for (int i = 0; i < dim; i++) {
				int state = datum.getStates()[i];

				if (state == -1) {
					states.append('?');
				} else {
					states.append(_variables[i].getStates().get(state));
				}

				if (i < dim - 1) {
					states.append(',');
				}
			}

			int nCopies = (int) datum.getWeight();
			for (int i = 0; i < nCopies; i++) {
				out.println(states);
			}
		}

		out.close();
	}

	/**
	 * Replaces the name of this data set.
	 * 
	 * @param name
	 *            new name of this data set.
	 */
	public void setName(String name) {
		name = name.trim();

		// name cannot be blank
		assert name.length() > 0;

		_name = name;
	}

	/**
	 * Returns a data set of the common variables shared by this data set and
	 * the specified BN. The new data set use the Variables in BayesNet.
	 * 
	 * @param bayesNet
	 *            BN on which a data set is to be returned.
	 * @return a data set of the common variables shared by this data set and
	 *         the specified BN.
	 * @suppressWarning("Unchecked")
	 */
	public DataSet synchronize(BayesNet bayesNet) {

		// common variables
		ArrayList<DiscreteVariable> variables = new ArrayList<DiscreteVariable>();
		// The map from a common Variable(in BN) to the index in the original
		// DataSet.
		HashMap<DiscreteVariable, Integer> map = new HashMap<DiscreteVariable, Integer>();

		for (int i = 0; i < getDimension(); i++) {
			for (DiscreteVariable variable : bayesNet.getDiscreteVariables()) {
				if (_variables[i].equals(variable)) {
					map.put(variable, i);
					variables.add(variable);
					break;
				}
			}
		}

		DataSet dataSet = new DataSet((DiscreteVariable[]) variables
				.toArray(new DiscreteVariable[variables.size()]));

		// projection
		int dimension = dataSet.getDimension();
		for (DataCase dataCase : _data) {
			int[] projectedStates = new int[dimension];
			int[] states = dataCase.getStates();
			for (int i = 0; i < dimension; i++) {
				projectedStates[i] = states[map.get(dataSet._variables[i])];
			}
			dataSet.addDataCase(projectedStates, dataCase.getWeight());
		}

		return dataSet;
	}

	/**
	 * Generate the training data of the given size. The original data becomes
	 * the testing data.
	 * 
	 * @param nOfTraining
	 * @return
	 */
	public DataSet splitIntoTrainingAndTesting(int nOfTraining) {
		DataSet training = new DataSet(_variables);
		for (int i = 0; i < nOfTraining; i++) {
			double total = getTotalWeight();
			double random = Uniform.staticNextDoubleFromTo(0.0, total);
			double accumulator = 0.0;
			// Find the datacase to be moved
			for (DataCase data : getData()) {
				double weight = data.getWeight();
				accumulator += weight;
				if (accumulator > random) {
					training.addDataCase(data.getStates(), 1.0);
					if (weight == 1.0) {
						getData().remove(data);
					} else {
						data.setWeight(weight - 1.0);
					}
					_totalWeight = _totalWeight - 1.0;
					break;
				}
			}
		}
		return training;
	}

	/**
	 * Creates a map from variable to its index used in this data set.
	 * 
	 * @return a map from variable to its index
	 */
	public Map<DiscreteVariable, Integer> createVariableToIndexMap() {
		return Algorithm.createIndexMap(_variables);
	}
}