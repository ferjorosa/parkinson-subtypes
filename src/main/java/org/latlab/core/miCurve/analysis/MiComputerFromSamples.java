package org.latlab.core.miCurve.analysis;//package org.latlab.core.miCurve;
//
//import java.io.PrintWriter;
//import java.util.Arrays;
//import java.util.List;
//
//import org.latlab.core.data.MixedDataSet;
//import org.latlab.core.util.DiscreteVariable;
//import org.latlab.core.util.Function;
//import org.latlab.core.util.Utils;
//import org.latlab.core.util.Variable;
//
//import weka.core.Instances;
//import weka.filters.Filter;
//import weka.filters.unsupervised.attribute.Discretize;
//
//public class MiComputerFromSamples {
//
//	/**
//	 * @param args
//	 * @throws Exception
//	 */
//	public static void main(String[] args) throws Exception {
//		if (args.length < 2) {
//			System.out.println("MiComputer data_file output_file");
//			return;
//		}
//
//		MixedDataSet data = MixedDataSet.read(args[0]);
//		data.setClassVariableToLast();
//
//		PrintWriter output = new PrintWriter(args[1]);
//
//		MiComputerFromSamples computer = new MiComputerFromSamples(data);
//		computer.compute(output);
//		output.close();
//
//	}
//
//	private final MixedDataSet samples;
//	private final List<Variable> variablesOrder;
//
//	public MiComputerFromSamples(MixedDataSet data) {
//		this.samples = data;
//		this.variablesOrder = data.getNonClassVariables();
//	}
//
//	public void compute(PrintWriter output) throws Exception {
//		computeGivenData(output);
//	}
//
//	public void computeGivenData(PrintWriter output) throws Exception {
//		MixedDataSet discretizedData = discretize(samples);
//		List<Variable> observedVariables = discretizedData.find(variablesOrder);
//		writeHeaderLine(output, observedVariables);
//
//		double[] mi =
//				computeEmpiricalMi(discretizedData,
//						discretizedData.getClassVariable(), observedVariables);
//		writeLine(output, "class", mi);
//	}
//
//	private void writeHeaderLine(PrintWriter output,
//			List<Variable> observedVariables) {
//		output.print("variable");
//		for (Variable variable : observedVariables) {
//			output.print("\t" + variable.getName());
//		}
//
//		output.println();
//	}
//
//	private void writeLine(PrintWriter output, String name, double[] mi) {
//		output.print(name);
//		for (double value : mi) {
//			output.print("\t" + value);
//		}
//
//		output.println();
//	}
//
//	private static MixedDataSet discretize(MixedDataSet data)
//			throws Exception {
//		Filter filter = new Discretize();
//		filter.setInputFormat(data.instances());
//		Instances instances = Filter.useFilter(data.instances(), filter);
//		return MixedDataSet.convert(instances);
//	}
//
//	private double[] computeEmpiricalMi(MixedDataSet discretizedData,
//			DiscreteVariable variable, List<Variable> observedVariables)
//			throws Exception {
//		double[] results = new double[observedVariables.size()];
//
//		for (int i = 0; i < results.length; i++) {
//			results[i] =
//					computeEmpiricalMi(discretizedData, variable,
//							(DiscreteVariable) observedVariables.get(i));
//		}
//
//		return results;
//	}
//
//	/**
//	 * Computes the mutual information between two discrete variables.
//	 * 
//	 * @param discretizedData
//	 * @param variable1
//	 * @param variable2
//	 * @return
//	 * @throws Exception
//	 */
//	private double computeEmpiricalMi(MixedDataSet discretizedData,
//			DiscreteVariable variable1, DiscreteVariable variable2)
//			throws Exception {
//		List<DiscreteVariable> variables = Arrays.asList(variable1, variable2);
//		int[] variableIndices = discretizedData.indicesOf(variables);
//
//		Function dist = Function.createFunction(variables);
//
//		for (int i = 0; i < discretizedData.size(); i++) {
//			List<Integer> states =
//					discretizedData.getStates(i, variableIndices);
//			dist.addToCell(variables, states, 1);
//		}
//
//		dist.normalize();
//
//		return Utils.computeNormalizedMutualInformation(dist);
//	}
//
//}
