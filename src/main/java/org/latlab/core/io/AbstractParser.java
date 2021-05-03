package org.latlab.core.io;

import org.latlab.core.model.BayesNet;

public abstract class AbstractParser implements Parser {
	public BayesNet parse() throws ParseException {
		BayesNet result = new BayesNet("");
		parse(result);
		return result;
	}
}
