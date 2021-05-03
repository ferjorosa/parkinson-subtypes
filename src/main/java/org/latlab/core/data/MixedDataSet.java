package org.latlab.core.data;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.jet.math.Functions;
import org.latlab.core.model.Gltm;
import org.latlab.core.reasoner.Evidences;
import org.latlab.core.util.*;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

public class MixedDataSet {

	private String name;
	private List<Variable> variables = new ArrayList<Variable>();
	private List<Instance> instances = new ArrayList<Instance>();

	private DoubleMatrix1D mean = null;
	private DoubleMatrix1D variance = null;
	private DoubleMatrix1D standardDeviation = null;
	private DoubleMatrix2D covariance = null;
	private double totalWeight = Double.NaN;
	private Evidences[] evidencesInstances = null;

	private int classIndex = -1;

	private String filename = null;

	/**
	 * Maps the variable to the index of the variable.
	 */
	private final Map<Variable, Integer> map;

	public static MixedDataSet createEmpty(List<Variable> variables, int capacity) {
		return new MixedDataSet(variables, capacity);
	}

	public MixedDataSet(Collection<Variable> variables, int capacity) {
		this("data", new ArrayList<Variable>(variables),
				new ArrayList<Instance>(capacity));
	}

	public MixedDataSet(String name, Collection<Variable> variables,
			Collection<Instance> instances) {
		this(name, new ArrayList<Variable>(variables),
				new ArrayList<Instance>(instances));
	}

	private MixedDataSet(String name, ArrayList<Variable> variables,
			ArrayList<Instance> instances) {
		this.name = name;
		this.variables = variables;
		this.instances = instances;
		this.map = Algorithm.createIndexMap(this.variables);
	}

	public String name() {
		return name;
	}

	public List<Variable> variables() {
		return variables;
	}

	public Instance get(int index) {
		return instances.get(index);
	}

	private void computeMeanAndCovariance() {
		// since the data may contain missing values, we can't use the
		// NormalSufficientStatistics to compute mean and covariance

		double[] m = computeMean();
		mean = DoubleFactory1D.dense.make(m);
		covariance = DoubleFactory2D.dense.make(computeCovariance(m));
	}

	private double[] computeMean() {
		double[] count = new double[variables.size()];
		double[] mean = new double[variables.size()];

		for (int j = 0; j < variables.size(); j++)
			count[j] = mean[j] = 0;

		for (Instance instance : instances) {
			for (int j = 0; j < variables.size(); j++) {
				if (!instance.isMissing(j)) {
					count[j] += instance.weight();
					mean[j] += instance.value(j) * instance.weight();
				}
			}
		}

		for (int j = 0; j < variables.size(); j++) {
			if (count[j] > 0)
				mean[j] = mean[j] / count[j];
			else
				mean[j] = 0;
		}

		return mean;
	}

	private double[][] computeCovariance(double[] mean) {
		double[][] count = new double[variables.size()][variables.size()];
		double[][] covariance = new double[variables.size()][variables.size()];

		for (int j = 0; j < variables.size(); j++)
			for (int k = 0; k < variables.size(); k++)
				count[j][k] = covariance[j][k] = 0;

		for (Instance instance : instances) {

			for (int j = 0; j < variables.size(); j++) {
				if (instance.isMissing(j))
					continue;

				for (int k = 0; k < variables.size(); k++) {
					if (instance.isMissing(k))
						continue;

					count[j][k] += instance.weight();
					covariance[j][k] += ((instance.value(j) - mean[j])
							* (instance.value(k) - mean[k])) * instance.weight();
				}
			}
		}

		for (int j = 0; j < variables.size(); j++) {
			for (int k = 0; k < variables.size(); k++) {
				if (count[j][k] > 0)
					covariance[j][k] = covariance[j][k] / (count[j][k]);
				else
					covariance[j][k] = 0;
			}
		}

		return covariance;
	}

	public DoubleMatrix1D mean() {
		if (mean == null)
			computeMeanAndCovariance();

		return mean;
	}

	/**
	 * The current implementation ignores all the instances that have a missing
	 * value.
	 * 
	 * @return
	 */
	public DoubleMatrix2D covariance() {
		if (covariance == null)
			computeMeanAndCovariance();

		return covariance;
	}

	public DoubleMatrix1D variance() {
		if (variance == null)
			variance = DoubleFactory2D.dense.diagonal(covariance());

		return variance;
	}

	public DoubleMatrix1D standardDeviation() {
		if (standardDeviation == null) {
			standardDeviation = variance().copy();
			standardDeviation.assign(Functions.sqrt);
		}

		return standardDeviation;
	}

	public int size() {
		return instances.size();
	}

	public double totalWeight() {
		if (Double.isNaN(totalWeight)) {
			totalWeight = totalWeight(0, size());
		}

		return totalWeight;
	}

	public double totalWeight(int start, int end) {
		totalWeight = 0;
		for (int i = start; i < end; i++) {
			totalWeight += get(i).weight();
		}

		return totalWeight;
	}

	public synchronized Evidences getEvidences(int index) {
		if (evidences()[index] == null) {
			evidences()[index] = convertToEvidences(get(index));
		}

		return evidences()[index];
	}

	private Evidences[] evidences() {
		if (evidencesInstances == null) {
			evidencesInstances = new Evidences[size()];
		}

		return evidencesInstances;
	}

	private Evidences convertToEvidences(Instance instance) {
		final Evidences evidences = new Evidences();

		for (int i = 0; i < variables.size(); i++) {
			if (instance.isMissing(i))
				continue;

			final double value = instance.value(i);
			variables.get(i).accept(new Variable.Visitor<Void>() {

				@Override
				public Void visit(DiscreteVariable variable) {
					evidences.add(variable, (int) value);
					return null;
				}

				@Override
				public Void visit(SingularContinuousVariable variable) {
					evidences.add(variable, value);
					return null;
				}

			});
		}

		return evidences;
	}

	/**
	 * Converts a instance into a vector form. Returns {@code null} if the
	 * instance contains any missing value.
	 * 
	 * @param instance
	 * @return
	 */
	private DoubleMatrix1D convertToVector(Instance instance) {
		return convertToVector(instance, false);
	}

	/**
	 * Converts a instance into a vector form. If {@code allowMissing} is true,
	 * missing values will be converted to NaN.
	 * 
	 * @param instance
	 * @return
	 */
	private DoubleMatrix1D convertToVector(Instance instance, boolean allowMissing) {
		DoubleMatrix1D vector = new DenseDoubleMatrix1D(variables.size());
		for (int i = 0; i < variables.size(); i++) {
			if (instance.isMissing(i)) {
				if (allowMissing)
					vector.setQuick(i, Double.NaN);
				else
					return null;
			} else
				vector.setQuick(i, instance.value(i));
		}

		return vector;
	}

	/**
	 * Synchronizes the variables of this data set with those in the model. It
	 * replaces the variables of this data set with those of the same names from
	 * the model.
	 * 
	 * <p>
	 * This function is necessary because the variables loaded from a file does
	 * not share the same set of variable instances as in this data, although
	 * they should refer to the same variables.
	 * 
	 * @param model
	 * @exception IllegalArgumentException
	 *                if variables with the same name do not have the same
	 *                cardinality or the same type
	 */
	public void synchronize(Gltm model) throws IllegalArgumentException {
		for (int i = 0; i < variables().size(); i++) {
			Variable attribute = variables.get(i);
			Variable variable = model.findVariableByName(attribute.getName());

			// if model does not contain this variable, skip it
			if (variable == null)
				continue;

			// check the variables
			if (!attribute.getClass().equals(variable.getClass())) {
				throw new IllegalArgumentException(String.format(
						"Variable [%s] has mismatched type.", variable.getName()));
			}

			if (attribute instanceof DiscreteVariable) {
				if (((DiscreteVariable) attribute)
						.getCardinality() != ((DiscreteVariable) variable)
								.getCardinality()) {
					throw new IllegalArgumentException(String.format(
							"Variable [%s] has mismatched number of states.",
							variable.getName()));
				}
			}

			Variable oldVariable = variables.get(i);
			map.put(variable, map.remove(oldVariable));
			variables.set(i, variable);
		}

		Evidences[] evidences = evidences();
		for (int i = 0; i < evidences.length; i++)
			evidences[i] = null;
	}

	public Integer indexOf(Variable variable) {
		return map.get(variable);
	}

	public int[] indicesOf(List<? extends Variable> variables) {
		int[] indices = new int[variables.size()];
		for (int i = 0; i < indices.length; i++) {
			indices[i] = indexOf(variables.get(i));
		}

		return indices;
	}

	public boolean hasClassVariable() {
		return classIndex >= 0;
	}

	public void setClassVariable(Variable variable) {
		int index = variable == null ? -1 : indexOf(variable);
		classIndex = index;
	}

	/**
	 * Sets the class variable to the last variable in this data set.
	 */
	public void setClassVariableToLast() {
		setClassVariable(variables().get(variables().size() - 1));
	}

	public DiscreteVariable getClassVariable() {
		return hasClassVariable() ? (DiscreteVariable) variables.get(classIndex) : null;
	}

	public int classIndex() {
		return classIndex;
	}

	public List<Variable> getNonClassVariables() {
		if (!hasClassVariable())
			return variables();

		return Algorithm.filter(variables(),
				new NotPredicate<Variable>(new ReferencePredicate(getClassVariable())));
	}

	public void removeMissingInstances() {
		List<Instance> noMissing = new ArrayList<Instance>(instances.size());

		for (Instance instance : instances) {
			if (!instance.hasMissing())
				noMissing.add(instance);
		}
	}

	public String filename() {
		return filename;
	}

	public void setFilename(String filename) {
		this.filename = filename;
	}

	/**
	 * Adds a new case to this data.
	 * 
	 * @param weight
	 * @param values
	 */
	public void add(double weight, double[] values) {
		instances.add(Instance.create(weight, values));
		clearCache();
	}

	private void clearCache() {
		mean = null;
		variance = null;
		covariance = null;
		standardDeviation = null;
		totalWeight = Double.NaN;
		evidencesInstances = null;
	}
}
