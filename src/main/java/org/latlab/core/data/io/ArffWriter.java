package org.latlab.core.data.io;

import org.latlab.core.data.Instance;
import org.latlab.core.data.MixedDataSet;
import org.latlab.core.util.DiscreteVariable;
import org.latlab.core.util.SingularContinuousVariable;
import org.latlab.core.util.Variable;

import java.io.*;
import java.util.List;

public class ArffWriter extends BaseWriter {

	PrintWriter writer;

	public static void write(String filename, MixedDataSet data)
			throws IOException {
		ArffWriter writer = new ArffWriter(new FileOutputStream(filename));
		writer.write(data);
	}

	public ArffWriter(String filename) throws UnsupportedEncodingException,
			FileNotFoundException {
		this(createWriter(filename));
	}

	public ArffWriter(OutputStream output) throws UnsupportedEncodingException,
			FileNotFoundException {
		this(createWriter(output));
	}

	public ArffWriter(Writer writer) {
		this.writer = new PrintWriter(writer);
	}

	public void write(MixedDataSet data) {
		writePreamble(data, data.name());
		writeInstances(data);

		writer.close();
	}

	// doesn't handle missing data!
	private void writeInstance(List<Variable> variables, Instance instance) {
		boolean first = true;
		for (int i = 0; i < variables.size(); i++) {
			if (!first)
				writer.print(",");
			else {
				first = false;
			}

			Variable variable = variables.get(i);
			if (instance.isMissing(i)) {
				writer.print("?");
			} else {
				double value = instance.value(i);
				if (variable instanceof DiscreteVariable) {
					writer.print(((DiscreteVariable) variable).getStates().get(
							(int) value));
				} else {
					writer.print(value);
				}
			}
		}

		if (instance.weight() != 1) {
			writer.printf(",{%f}", instance.weight());
		}

		writer.println();
	}

	private void writeInstances(MixedDataSet data) {
		writer.println("@data");
		for (int i = 0; i < data.size(); i++) {
			writeInstance(data.variables(), data.get(i));
		}
	}

	private void writePreamble(MixedDataSet data, String relationName) {
		writer.printf("@relation %s\n", relationName);
		writeVariables(data.variables());
	}

	private void writeVariables(List<Variable> variables) {
		for (Variable variable : variables) {
			if (variable instanceof DiscreteVariable) {
				writer.printf("@attribute %s {%s}\n", variable.getName(),
						getNominalSpecification((DiscreteVariable) variable));

				// writer.println("@attribute "+variable.name()+" numeric");
			} else if (variable instanceof SingularContinuousVariable) {
				writer.printf("@attribute %s REAL\n", variable.getName());
			} else
				throw new IllegalArgumentException(
						"Data contains non-nominal variables, which are not supported at this moment.");
		}
	}

	private String getNominalSpecification(DiscreteVariable variable) {
		StringBuilder builder = new StringBuilder();

		for (String string : variable.getStates()) {
			builder.append(string).append(",");
		}

		// delete the last comma
		builder.deleteCharAt(builder.length() - 1);

		return builder.toString();
	}
}