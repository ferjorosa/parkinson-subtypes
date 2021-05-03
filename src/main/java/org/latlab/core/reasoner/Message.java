package org.latlab.core.reasoner;

import org.latlab.core.util.Function;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Represents the messages passed in propagation.
 * 
 * @author leonard
 * 
 */
public class Message {
	/**
	 * The potential passed.
	 */
	public final Function function;

	/**
	 * Log of the constant used for normalizing the potential.
	 */
	private double logNormalization;

	public Message(Function function, double logNormalization) {
		this.function = function;
		this.logNormalization = logNormalization;
	}

	public Message clone() {
		return new Message(function.clone(), logNormalization);
	}

	public double logNormalization() {
		return logNormalization;
	}

	public Message times(Message multiplier) {
		return new Message(function.times(multiplier.function),
				logNormalization + multiplier.logNormalization);
	}

	public static Message computeProduct(Collection<Message> messages) {
		List<Function> functions = new ArrayList<Function>(messages.size());
		double logProduct = 0;

		for (Message message : messages) {
			functions.add(message.function);
			logProduct += message.logNormalization;
		}

		return new Message(Function.computeProduct(functions), logProduct);
	}

	public void divide(Message divider) {
		function.divide(divider.function);
		logNormalization -= divider.logNormalization;
	}

	public void divide(Function divider) {
		function.divide(divider);
	}
}
