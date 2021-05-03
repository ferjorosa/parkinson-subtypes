package org.latlab.core.data.io.arff;

import org.latlab.core.data.MixedDataSet;

import java.io.FileInputStream;
import java.io.IOException;

public class ArffLoader {
	public static MixedDataSet load(String path) throws IOException,
			ParseException {
		MixedDataSet data = ArffParser.parse(new FileInputStream(path));
		data.setFilename(path);
		return data;
	}
}
