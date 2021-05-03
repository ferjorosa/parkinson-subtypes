package org.latlab.core.miCurve.analysis;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

/**
 * Used to read classification files. A classification file holds a table
 * listing the posterior probabilities of the states for each data case in each
 * row. The first row lists the state names, and the first column lists the data
 * case index.
 * 
 * @author leonard
 * 
 */
public class ClassificationFileReader {

	private final String filename;
	private final int length;

	private List<String> header;
	private List<List<Double>> content;

	public ClassificationFileReader(String filename, int length)
			throws IOException {
		this.filename = filename;
		this.length = length;

		read();
	}

	private void read() throws IOException {

		BufferedReader reader = new BufferedReader(new FileReader(filename));

		// skip the first line, which is the header line
		String line = reader.readLine();
		readHeader(line);

		content = new ArrayList<List<Double>>(length);

		while (reader.ready()) {
			line = reader.readLine();
			StringTokenizer tokenizer = new StringTokenizer(line, ",");

			// skip also the first column, which indicates the data index
			// (added by R)
			tokenizer.nextToken();

			List<Double> values =
					new ArrayList<Double>(tokenizer.countTokens());

			while (tokenizer.hasMoreTokens()) {
				String token = tokenizer.nextToken();
				values.add(Double.parseDouble(token));
			}

			content.add(values);
		}

	}

	private void readHeader(String line) {
		StringTokenizer tokenizer = new StringTokenizer(line, ",");

		// skip also the first column, which indicates the data index
		// (added by R)
		tokenizer.nextToken();

		header = new ArrayList<String>(tokenizer.countTokens());

		while (tokenizer.hasMoreTokens()) {
			String token = tokenizer.nextToken();

			if (token.startsWith("\"") && token.endsWith("\"")) {
				token = token.substring(1, token.length() - 1);
			}

			header.add(token);
		}
	}

	public List<String> header() {
		return header;
	}

	public List<List<Double>> values() {
		return content;
	}
}
