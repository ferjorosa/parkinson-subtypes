package org.latlab.core.miCurve.analysis;

import org.latlab.core.data.MixedDataSet;
import org.latlab.core.data.io.arff.ArffLoader;
import org.latlab.core.io.bif.BifParser;
import org.latlab.core.model.Gltm;
import org.latlab.core.reasoner.NaturalCliqueTreePropagation;
import org.latlab.core.util.DiscreteVariable;
import org.latlab.core.util.Function;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.*;

/**
 * Computes the classification probabilities for each latent variable.
 * 
 * @author leonard
 * 
 */
public class ClassificationComputer {

	/**
	 * @param args
	 * @throws Exception
	 * @throws FileNotFoundException
	 */
	public static void main(String[] args) throws Exception {
		if (args.length < 3) {
			System.out.println(
					"ClassificationComputer " + "model_file data_file output_prefix");
			return;
		}
		
		run(args[0], args[1], args[2], false);
	}

	public static void run(String modelFile, String dataFile, String outputPrefix,
			boolean allowMissing) throws Exception {
		MixedDataSet data = ArffLoader.load(dataFile);
		data.setClassVariableToLast();

		if (!allowMissing)
			data.removeMissingInstances();

		Gltm model = new BifParser(new FileInputStream(modelFile), "UTF-8")
				.parse(new Gltm());
		data.synchronize(model);

		ClassificationComputer computer = new ClassificationComputer(model, data,
				outputPrefix);
		computer.compute();
		computer.computeForTarget();
	}

	private final Gltm model;
	private final MixedDataSet data;
	private final String prefix;
	private final NaturalCliqueTreePropagation propagation;
	private final Map<DiscreteVariable, PrintWriter> outputs;

	public ClassificationComputer(Gltm model, MixedDataSet data, String prefix) {
		this.model = model;
		this.data = data;
		this.prefix = prefix;
		this.propagation = new NaturalCliqueTreePropagation(model);
		outputs = new HashMap<DiscreteVariable, PrintWriter>();
	}

	public void compute() throws FileNotFoundException {
		Set<DiscreteVariable> latentVariables = model.getInternalVars();
		for (DiscreteVariable latent : latentVariables) {
			PrintWriter output = openFile(latent);
			outputs.put(latent, output);
			writeHeader(latent, output);
		}

		for (int i = 0; i < data.size(); i++) {
			propagation.use(data.getEvidences(i));
			propagation.propagate();

			for (DiscreteVariable latent : latentVariables) {
				Function probabilities = propagation.getMarginal(latent);
				writeLine(i, probabilities, outputs.get(latent));
			}
		}

		for (DiscreteVariable latent : latentVariables) {
			outputs.get(latent).close();
		}
	}

	public void computeForTarget() throws FileNotFoundException {
		DiscreteVariable classVariable = data.getClassVariable();
		int classIndex = data.indexOf(classVariable);

		PrintWriter output = openFile(classVariable);

		writeHeader(classVariable, output);

		for (int i = 0; i < data.size(); i++) {
			double state = data.get(i).value(classIndex);
			Function probabilities = Function.createIndicatorFunction(classVariable,
					(int) state);
			writeLine(i, probabilities, output);

		}

		output.close();
	}

	private PrintWriter openFile(DiscreteVariable latent) throws FileNotFoundException {
		String filename = prefix + "." + latent.getName() + ".classification.csv";
		return new PrintWriter(filename);
	}

	private void writeHeader(DiscreteVariable latent, PrintWriter output) {
		List<String> values = new ArrayList<String>(latent.getCardinality() + 1);

		values.add("\"\"");
		for (String state : latent.getStates()) {
			values.add("\"" + state + "\"");
		}

		write(values, output, ",");
	}

	private void writeLine(int index, Function probabilities, PrintWriter output) {
		double[] cells = probabilities.getCells();

		List<String> values = new ArrayList<String>(cells.length + 1);
		values.add("\"" + index + "\"");

		for (double cell : cells) {
			values.add(Double.toString(cell));
		}

		write(values, output, ",");
	}

	private void write(List<String> values, PrintWriter output, String separator) {
		for (int i = 0; i < values.size(); i++) {
			output.write(values.get(i));

			if (i < values.size() - 1) {
				output.print(separator);
			}
		}

		output.println();
	}
}
