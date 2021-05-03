package org.latlab.core.miCurve.analysis;//package org.latlab.core.miCurve;
//
//import java.io.BufferedReader;
//import java.io.FileReader;
//import java.io.IOException;
//import java.io.PrintWriter;
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.Collections;
//import java.util.List;
//import java.util.StringTokenizer;
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
//public class FeatureProfileComputer {
//
//	enum Type {
//		NMI, ANMI, MI
//	};
//
//	/**
//	 * @param args
//	 * @throws Exception
//	 */
//	public static void main(String[] args) throws Exception {
//		if (args.length < 2) {
//			System.out.println("FeatureProfileComputer "
//					+ "[-{nmi|anmi|mi}] data_file "
//					+ "classification_file [output_file]");
//			return;
//		}
//
//		int arg = 0;
//
//		Type type = Type.NMI;
//		if (args[arg].equalsIgnoreCase("-NMI")) {
//			type = Type.NMI;
//			arg++;
//		} else if (args[arg].equalsIgnoreCase("-ANMI")) {
//			type = Type.ANMI;
//			arg++;
//		} else if (args[arg].equalsIgnoreCase("-MI")) {
//			type = Type.MI;
//			arg++;
//		}
//
//		MixedDataSet data = MixedDataSet.read(args[arg]);
//		data.setClassVariableToLast();
//		arg++;
//
//		String classFile = args[arg];
//		List<List<Double>> classification =
//				readProbabilities(classFile, data.size());
//		arg++;
//
//		String outputFileName =
//				args.length > arg ? args[arg] : getOutputFileName(classFile,
//						type);
//		PrintWriter output = new PrintWriter(outputFileName);
//
//		FeatureProfileComputer computer =
//				new FeatureProfileComputer(data, classification, type);
//		computer.compute(output);
//		output.close();
//	}
//
//	private static String getOutputFileName(String classFile, Type type) {
//		if (type == Type.NMI) {
//			return classFile + ".nmi.csv";
//		} else if (type == Type.MI) {
//			return classFile + ".mi.csv";
//		} else if (type == Type.ANMI) {
//			return classFile + ".anmi.csv";
//		} else {
//			return classFile + ".nmi.csv";
//		}
//	}
//
//	private static List<List<Double>> readProbabilities(String filename,
//			int length) throws IOException {
//		List<List<Double>> result = new ArrayList<List<Double>>(length);
//
//		BufferedReader reader = new BufferedReader(new FileReader(filename));
//
//		// skip the first line, which is the header line
//		String line = reader.readLine();
//
//		while (reader.ready()) {
//			line = reader.readLine();
//			StringTokenizer tokenizer = new StringTokenizer(line, ",");
//
//			// skip also the first column, which indicates the data index
//			// (added by R)
//			tokenizer.nextToken();
//
//			List<Double> values =
//					new ArrayList<Double>(tokenizer.countTokens());
//
//			while (tokenizer.hasMoreTokens()) {
//				String token = tokenizer.nextToken();
//				values.add(Double.parseDouble(token));
//			}
//
//			result.add(values);
//		}
//
//		return result;
//	}
//
//	private final MixedDataSet discretizedData;
//	private final DiscreteVariable latentVariable;
//	private final List<List<Double>> classification;
//	private final Type type;
//
//	public FeatureProfileComputer(MixedDataSet data,
//			List<List<Double>> classification, Type type) throws Exception {
//		this.latentVariable =
//				new DiscreteVariable(classification.get(0).size());
//		this.classification = classification;
//		discretizedData = discretize(data);
//		this.type = type;
//	}
//
//	private void compute(PrintWriter output) {
//		writeHeaderLine(output, discretizedData.getNonClassVariables());
//		List<Double> values = computeEmpiricalValues();
//		writeLine(output, "class", values);
//
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
//	private void writeLine(PrintWriter output, String name, List<Double> values) {
//		output.print(name);
//		for (double value : values) {
//			output.print("\t" + value);
//		}
//
//		output.println();
//	}
//
//	private List<Double> computeEmpiricalValues() {
//		List<Double> values =
//				new ArrayList<Double>(
//						discretizedData.getNonClassVariables().size());
//
//		for (Variable variable : discretizedData.getNonClassVariables()) {
//			Function empirical =
//					computeEmpiricalJointProbability((DiscreteVariable) variable);
//
//			if (type == Type.NMI) {
//				values.add(Utils.computeNormalizedMutualInformation(empirical));
//			} else if (type == Type.MI) {
//				values.add(Utils.computeMutualInformation(empirical));
//			} else if (type == Type.ANMI) {
//				values.add(Utils.computeAsymmetricNMI(empirical));
//			} else {
//				values.add(Utils.computeNormalizedMutualInformation(empirical));
//			}
//		}
//
//		return values;
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
//	private Function computeEmpiricalJointProbability(DiscreteVariable variable) {
//		List<DiscreteVariable> variables =
//				Arrays.asList(latentVariable, variable);
//
//		Function joint = Function.createFunction(variables);
//		joint.times(0.0);
//
//		List<DiscreteVariable> latentVariableList =
//				Collections.singletonList(latentVariable);
//
//		int variableIndex = discretizedData.indexOf(variable);
//
//		// sum_{ d in D} P(latents|d) P(C|d) * P(d) where P(d) is assumed to be
//		// 1/n
//		for (int i = 0; i < discretizedData.size(); i++) {
//			int state = (int) discretizedData.get(i).value(variableIndex);
//
//			Function classProbability =
//					Function.createFunction(latentVariableList);
//			classProbability.setCells(latentVariableList, classification.get(i));
//
//			Function observedProbability =
//					Function.createIndicatorFunction(variable, state);
//
//			joint.plus(classProbability.times(observedProbability));
//		}
//
//		joint.normalize();
//
//		return joint;
//	}
//}
