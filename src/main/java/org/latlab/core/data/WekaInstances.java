package org.latlab.core.data;//package org.latlab.core.data;
//
//import java.io.File;
//import java.io.FileInputStream;
//import java.io.IOException;
//import java.io.InputStream;
//import java.io.InputStreamReader;
//import java.util.ArrayList;
//import java.util.Collection;
//import java.util.Collections;
//import java.util.HashMap;
//import java.util.List;
//import java.util.Map;
//
//import org.latlab.core.learner.geast.NormalSufficientStatistics;
//import org.latlab.core.model.Gltm;
//import org.latlab.core.model.VariableNamePredicate;
//import org.latlab.core.reasoner.Evidences;
//import org.latlab.core.util.Algorithm;
//import org.latlab.core.util.DiscreteVariable;
//import org.latlab.core.util.JointContinuousVariable;
//import org.latlab.core.util.NotPredicate;
//import org.latlab.core.util.ReferencePredicate;
//import org.latlab.core.util.SingularContinuousVariable;
//import org.latlab.core.util.Variable;
//import org.latlab.core.util.Variable.Visitor;
//
//import weka.core.Attribute;
//import weka.core.FastVector;
//import weka.core.Instance;
//import weka.core.Instances;
//import weka.core.converters.ArffLoader;
//import weka.core.converters.ArffSaver;
//import cern.colt.matrix.DoubleFactory2D;
//import cern.colt.matrix.DoubleMatrix1D;
//import cern.colt.matrix.DoubleMatrix2D;
//import cern.colt.matrix.impl.DenseDoubleMatrix1D;
//import cern.jet.math.Functions;
//
//public class WekaInstances {
//
//	private static Attribute convert(Variable variable) {
//		return variable.accept(new Variable.Visitor<Attribute>() {
//
//			@Override
//			public Attribute visit(DiscreteVariable variable) {
//				FastVector states = new FastVector(variable.getCardinality());
//				for (String state : variable.getStates())
//					states.addElement(state);
//				return new Attribute(variable.getName(), states);
//			}
//
//			@Override
//			public Attribute visit(JointContinuousVariable variable) {
//				throw new IllegalArgumentException();
//			}
//
//			@Override
//			public Attribute visit(SingularContinuousVariable variable) {
//				return new Attribute(variable.getName());
//			}
//		});
//	}
//
//	/**
//	 * Used to select a few instances of the whole data set.
//	 * 
//	 * @author leonard
//	 * @see WekaInstances#select(int[])
//	 * 
//	 */
//	private class Selection extends WekaInstances {
//		private final int[] selection;
//
//		public Selection(int[] selection) {
//			// uses the fields of the outer class (not super class, as those
//			// fields are private) as argument
//			// to the constructor
//			super(instances, variables, map, evidencesInstances);
//			this.selection = selection;
//		}
//
//		private int convert(int index) {
//			return selection[index];
//		}
//
//		@Override
//		public int size() {
//			return selection.length;
//		}
//
//		@Override
//		public Instance get(int index) {
//			return super.get(convert(index));
//		}
//
//		@Override
//		public Evidences getEvidences(int index) {
//			return super.getEvidences(convert(index));
//		}
//	}
//
//	private final List<Instance> instances;
//
//	/**
//	 * Holds the list of variables, each of which corresponds to an attribute in
//	 * the data. The indices of the variables are the same as those of the
//	 * attributes in the data.
//	 */
//	private final List<Variable> variables;
//
//	/**
//	 * Maps the variable to the index of the variable.
//	 */
//	private final Map<Variable, Integer> map;
//
//	private DoubleMatrix1D mean = null;
//	private DoubleMatrix1D variance = null;
//	private DoubleMatrix1D standardDeviation = null;
//	private DoubleMatrix2D covariance = null;
//	private double totalWeight = Double.NaN;
//	private Evidences[] evidencesInstances = null;
//
//	private String filename = "";
//
//	private WekaInstances(Instances instances) {
//		this.instances = instances;
//
//		variables = new ArrayList<Variable>(instances.numAttributes());
//		map = new HashMap<Variable, Integer>(instances.numAttributes());
//
//		for (int i = 0; i < this.instances.numAttributes(); i++) {
//			Variable variable = convert(instances.attribute(i));
//			map.put(variable, i);
//			variables.add(variable);
//		}
//	}
//
//	private WekaInstances(List<Variable> variables, int capacity) {
//		this.variables = new ArrayList<Variable>(variables);
//		map = new HashMap<Variable, Integer>(variables.size());
//
//		FastVector attributes = new FastVector(variables.size());
//
//		for (int i = 0; i < variables.size(); i++) {
//			Variable variable = variables.get(i);
//			attributes.addElement(convert(variable));
//
//			map.put(variable, i);
//		}
//
//		instances = new Instances("Data", attributes, capacity);
//	}
//
//	private WekaInstances(Instances instances, List<Variable> variables,
//			Map<Variable, Integer> map, Evidences[] evidences) {
//		this.instances = instances;
//		this.variables = variables;
//		this.map = map;
//		this.evidencesInstances = evidences;
//	}
//
//	public static WekaInstances read(String filename) throws IOException {
//		return read(new FileInputStream(filename), filename);
//	}
//
//	public static WekaInstances read(InputStream stream, String filename)
//			throws IOException {
//		WekaInstances data = read(new FileInputStream(filename));
//		data.setFilename(filename);
//		return data;
//	}
//
//	public static WekaInstances read(InputStream stream) throws IOException {
//		ArffLoader.ArffReader reader =
//				new ArffLoader.ArffReader(
//						new InputStreamReader(stream, "UTF-8"));
//		return new WekaInstances(reader.getData());
//	}
//
//	public static WekaInstances convert(Instances instances) {
//		return new WekaInstances(instances);
//	}
//
//	public void setFilename(String filename) {
//		this.filename = filename;
//	}
//
//	public String filename() {
//		return filename;
//	}
//
//	public String name() {
//		return instances().relationName();
//	}
//
//	public static WekaInstances createEmpty(List<Variable> variables,
//			int capacity) {
//		return new WekaInstances(variables, capacity);
//	}
//
//	/**
//	 * Selects some instances and returns a subset view of this data set.
//	 * 
//	 * @param selection
//	 *            indices of the selected instances
//	 * @return subset view of this data set
//	 */
//	public WekaInstances select(int[] selection) {
//		return new Selection(selection);
//	}
//
//	private Variable convert(Attribute attribute) {
//		if (attribute.isNominal()) {
//			return convertDiscreteVariable(attribute);
//		} else if (attribute.isNumeric()) {
//			return convertContinuousVariable(attribute);
//		} else {
//			throw new IllegalArgumentException(
//					"Unsupported attribute type from the data.");
//		}
//	}
//
//	private DiscreteVariable convertDiscreteVariable(Attribute attribute) {
//		List<String> states = new ArrayList<String>(attribute.numValues());
//		for (int i = 0; i < attribute.numValues(); i++) {
//			states.add(attribute.value(i));
//		}
//
//		return new DiscreteVariable(attribute.name(), states);
//	}
//
//	private SingularContinuousVariable convertContinuousVariable(
//			Attribute attribute) {
//		return new SingularContinuousVariable(attribute.name());
//	}
//
//	public List<Variable> variables() {
//		return Collections.unmodifiableList(variables);
//	}
//
//	public boolean hasClassVariable() {
//		return instances.classIndex() >= 0;
//	}
//
//	public void setClassVariable(Variable variable) {
//		int index = variable == null ? -1 : map.get(variable);
//		instances.setClassIndex(index);
//	}
//
//	/**
//	 * Sets the class variable to the last variable in this data set.
//	 */
//	public void setClassVariableToLast() {
//		setClassVariable(variables().get(variables().size() - 1));
//	}
//
//	public DiscreteVariable getClassVariable() {
//		return hasClassVariable() ? (DiscreteVariable) variables.get(instances.classIndex())
//				: null;
//	}
//
//	public List<Variable> getNonClassVariables() {
//		if (!hasClassVariable())
//			return variables();
//
//		return Algorithm.filter(variables(), new NotPredicate<Variable>(
//				new ReferencePredicate(getClassVariable())));
//	}
//
//	public void remove(Variable variable) {
//		final int index = variables.indexOf(variable);
//
//		instances.deleteAttributeAt(index);
//
//		variables.remove(index);
//		map.remove(variable);
//
//		// shift the indices for those variables after the removed one
//		for (Map.Entry<Variable, Integer> entry : map.entrySet()) {
//			if (entry.getValue() > index) {
//				entry.setValue(entry.getValue() - 1);
//			}
//		}
//	}
//
//	public void removeMissingInstances() {
//		for (int i = 0; i < instances.numAttributes(); i++) {
//			instances.deleteWithMissing(i);
//		}
//	}
//
//	public int size() {
//		return instances.numInstances();
//	}
//
//	public Instance get(int index) {
//		return instances.instance(index);
//	}
//
//	public double getValue(int index, Variable variable) {
//		return instances.instance(index).value(indexOf(variable));
//	}
//
//	public Integer indexOf(Variable variable) {
//		return map.get(variable);
//	}
//
//	public int[] indicesOf(List<? extends Variable> variables) {
//		int[] indices = new int[variables.size()];
//		for (int i = 0; i < indices.length; i++) {
//			indices[i] = indexOf(variables.get(i));
//		}
//
//		return indices;
//	}
//
//	public synchronized Evidences getEvidences(int index) {
//		if (evidences()[index] == null) {
//			evidences()[index] = convertToEvidences(instances.instance(index));
//		}
//
//		return evidences()[index];
//	}
//
//	private Evidences[] evidences() {
//		if (evidencesInstances == null) {
//			evidencesInstances = new Evidences[instances.numInstances()];
//		}
//
//		return evidencesInstances;
//	}
//
//	private Evidences convertToEvidences(Instance instance) {
//		final Evidences evidences = new Evidences();
//
//		for (int i = 0; i < variables.size(); i++) {
//			if (instance.isMissing(i))
//				continue;
//
//			final double value = instance.value(i);
//			variables.get(i).accept(new Variable.Visitor<Void>() {
//
//				@Override
//				public Void visit(DiscreteVariable variable) {
//					evidences.add(variable, (int) value);
//					return null;
//				}
//
//				@Override
//				public Void visit(SingularContinuousVariable variable) {
//					evidences.add(variable, value);
//					return null;
//				}
//
//			});
//		}
//
//		return evidences;
//	}
//
//	public Instances instances() {
//		return instances;
//	}
//
//	/**
//	 * Returns the states of the variables for the data case with specified
//	 * index.
//	 * 
//	 * @param index
//	 *            index of the data case
//	 * @param variableIndices
//	 *            indices of variables of interest
//	 * @return states of the variables
//	 */
//	public List<Integer> getStates(int index, int[] variableIndices) {
//		Instance instance = get(index);
//		List<Integer> states = new ArrayList<Integer>(variables.size());
//
//		for (int variableIndex : variableIndices) {
//			states.add((int) instance.value(variableIndex));
//		}
//
//		return states;
//	}
//
//	public DoubleMatrix1D mean() {
//		if (mean == null)
//			computeMeanAndCovariance();
//
//		return mean;
//	}
//
//	/**
//	 * The current implementation ignores all the instances that have a missing
//	 * value.
//	 * 
//	 * @return
//	 */
//	public DoubleMatrix2D covariance() {
//		if (covariance == null)
//			computeMeanAndCovariance();
//
//		return covariance;
//	}
//
//	public DoubleMatrix1D variance() {
//		if (variance == null)
//			variance = DoubleFactory2D.dense.diagonal(covariance());
//
//		return variance;
//	}
//
//	public DoubleMatrix1D standardDeviation() {
//		if (standardDeviation == null) {
//			standardDeviation = variance().copy();
//			standardDeviation.assign(Functions.sqrt);
//		}
//
//		return standardDeviation;
//	}
//
//	public double totalWeight() {
//		if (Double.isNaN(totalWeight)) {
//			totalWeight = totalWeight(0, size());
//		}
//
//		return totalWeight;
//	}
//
//	public double totalWeight(int start, int end) {
//		totalWeight = 0;
//		for (int i = start; i < end; i++) {
//			totalWeight += get(i).weight();
//		}
//
//		return totalWeight;
//	}
//
//	private void computeMeanAndCovariance() {
//		NormalSufficientStatistics statistics =
//				new NormalSufficientStatistics(variables.size());
//
//		for (int i = 0; i < size(); i++) {
//			Instance instance = get(i);
//			statistics.add(convertToVector(instance), instance.weight());
//		}
//
//		mean = statistics.computeMean();
//		covariance = statistics.computeCovariance();
//	}
//
//	/**
//	 * Converts a instance into a vector form. Returns {@code null} if the
//	 * instance contains any missing value.
//	 * 
//	 * @param instance
//	 * @return
//	 */
//	private DoubleMatrix1D convertToVector(Instance instance) {
//		DoubleMatrix1D vector =
//				new DenseDoubleMatrix1D(instance.numAttributes());
//		for (int i = 0; i < instance.numAttributes(); i++) {
//			if (instance.isMissing(i))
//				return null;
//			vector.setQuick(i, instance.value(i));
//		}
//
//		return vector;
//	}
//
//	/**
//	 * Synchronizes the variables of this data set with those in the model. It
//	 * replaces the variables of this data set with those of the same names from
//	 * the model.
//	 * 
//	 * <p>
//	 * This function is necessary because the variables loaded from a file does
//	 * not share the same set of variable instances as in this data, although
//	 * they should refer to the same variables.
//	 * 
//	 * @param model
//	 * @exception IllegalArgumentException
//	 *                if variables with the same name do not have the same
//	 *                cardinality or the same type
//	 */
//	public void synchronize(Gltm model) throws IllegalArgumentException {
//		for (int i = 0; i < instances.numAttributes(); i++) {
//			Attribute attribute = instances.attribute(i);
//			Variable variable = model.findVariableByName(attribute.name());
//
//			// if model does not contain this variable, skip it
//			if (variable == null)
//				continue;
//
//			// check the variables
//			if ((attribute.isNominal() && !(variable instanceof DiscreteVariable))
//					|| (attribute.isNumeric() && !(variable instanceof SingularContinuousVariable))) {
//				throw new IllegalArgumentException(String.format(
//						"Variable [%s] has mismatched type.",
//						variable.getName()));
//			}
//
//			if (attribute.isNominal()) {
//				if (attribute.numValues() != ((DiscreteVariable) variable).getCardinality()) {
//					throw new IllegalArgumentException(String.format(
//							"Variable [%s] has mismatched number of states.",
//							variable.getName()));
//				}
//			}
//
//			Variable oldVariable = variables.get(i);
//			map.put(variable, map.remove(oldVariable));
//			variables.set(i, variable);
//		}
//
//		Evidences[] evidences = evidences();
//		for (int i = 0; i < evidences.length; i++)
//			evidences[i] = null;
//	}
//
//	/**
//	 * Adds a new case to this data.
//	 * 
//	 * @param weight
//	 * @param values
//	 */
//	public void add(double weight, double[] values) {
//		instances.add(new Instance(weight, values));
//		clearCache();
//	}
//
//	public Variable find(String name) {
//		return Algorithm.linearSearch(variables,
//				new VariableNamePredicate(name));
//	}
//
//	/**
//	 * Finds the variables in this data set that matches the names of the given
//	 * list of variables.
//	 * 
//	 * @param <T>
//	 *            type of variables
//	 * @param variables
//	 *            of which the names to match
//	 * @return variables having same names in this data set as the given
//	 *         variables
//	 */
//	@SuppressWarnings("unchecked")
//	public <T extends Variable> List<T> find(Collection<T> variables) {
//		List<T> results = new ArrayList<T>(variables.size());
//
//		for (T v : variables) {
//			results.add((T) find(v.getName()));
//		}
//
//		return results;
//	}
//
//	private void clearCache() {
//		mean = null;
//		variance = null;
//		covariance = null;
//		totalWeight = Double.NaN;
//		evidencesInstances = null;
//	}
//
//	public void write(String filename) throws IOException {
//		ArffSaver saver = new ArffSaver();
//		saver.setInstances(instances);
//		saver.setFile(new File(filename));
//
//		saver.writeBatch();
//	}
//}
