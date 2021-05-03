package org.latlab.core.reasoner;

import org.latlab.core.graph.AbstractNode;
import org.latlab.core.util.DiscreteVariable;
import org.latlab.core.util.Potential;
import org.latlab.core.util.Variable;

import java.util.Set;

/**
 * A clique in a clique tree for a Bayesian network.
 * 
 * @author leonard
 * 
 */
public abstract class Clique extends CliqueTreeNode {

	/**
	 * Used to visit the neighboring cliques. This visitor provides the neighbor
	 * cliques and the separators between this clique the the neighbor cliques.
	 * 
	 * <p>
	 * Note that since the neighboring nodes of a clique in a clique tree should
	 * be a separator, the use of this visitor may simplify the process to
	 * enumerate the neighboring cliques.
	 * 
	 * @author leonard
	 * 
	 */
	public static abstract class NeighborVisitor {
		private Separator origin;

		/**
		 * Constructs this visitor by specifying an origin of visit. The origin
		 * separator is skipped when enumerating the neighbor separators in the
		 * visit.
		 * 
		 * @param origin
		 *            origin of visit, or {@code null} if all separators should
		 *            be visited
		 */
		protected NeighborVisitor(Separator origin) {
			this.origin = origin;
		}

		public Separator origin() {
			return origin;
		}

		public abstract void visit(Separator separator, Clique neighbor);
	}

	private boolean focus = true;
	protected boolean pivot = false;

	protected Clique(NaturalCliqueTree tree, String name) {
		super(tree, name);
	}

	public abstract boolean contains(Variable variable);

	/**
	 * Visits the neighboring cliques (not separators).
	 * 
	 * @param visitor
	 *            used to visit the neighboring cliques
	 */
	public void visitNeighbors(NeighborVisitor visitor) {
		for (AbstractNode separator : getNeighbors()) {
			if (separator == visitor.origin())
				continue;

			for (AbstractNode neighbor : separator.getNeighbors()) {
				if (neighbor != this) {
					visitor.visit((Separator) separator, (Clique) neighbor);
				}
			}
		}
	}

	/**
	 * Computes the message for sending to the neighbor {@code separator}.
	 * 
	 * @param separator
	 *            neighbor of this clique to send message to
	 * @return message sending to {@code separator}
	 */
	public Message computeMessage(Separator separator) {
		return computeMessage(null, separator, null);
	}

	/**
	 * Computes the message for sending to the neighbor {@code separator}, by
	 * first multiplying the potential by the {@code multiplier}, and retains
	 * the given {@code retainingVariables} and does not marginalize out them.
	 * 
	 * @param multiplier
	 *            multiplied to the potential before the message is computed, or
	 *            {@code null} if not multiplication is need
	 * @param separator
	 *            neighbor of this clique to send message to
	 * @param retainingVariables
	 *            variables to be retained in the function. If this is {@code
	 *            null}, it does not retain any additional variables.
	 * @return message sending to {@code separator}
	 */
	public abstract Message computeMessage(Message multiplier,
			Separator separator, Set<DiscreteVariable> retainingVariables);

	/**
	 * Combines the potential on this clique with another potential. It
	 * normalizes the potential on this clique after combination.
	 * 
	 * @param other
	 *            potential to combine with the potential on this clique.
	 * @param logNormalization
	 *            log of normalization constant
	 */
	public abstract void combine(Potential other, double logNormalization);

	public void combine(Potential other) {
		combine(other, 0);
	}

	public void combine(Message message) {
		combine(message.function, message.logNormalization());
	}

	public void normalize(double constant) {
		double value = potential().normalize(constant);
		addLogNormalization(Math.log(value));
	}

	public abstract double logNormalization();
	
	protected abstract void addLogNormalization(double logNormalization);

	public void setFocus(boolean value) {
		focus = value;
	}

	public boolean focus() {
		return focus;
	}

	/**
	 * Sets whether this clique is used as the pivot in propagation. It affects
	 * the {@code combine} operation.
	 * 
	 * @param pivot
	 *            whether it is the pivot clique
	 */
	public void setPivot(boolean pivot) {
		this.pivot = pivot;
	}

	@Override
	public String toString(int amount) {
		// amount must be non-negative
		assert amount >= 0;

		// prepares white space for indent
		StringBuffer whiteSpace = new StringBuffer();
		for (int i = 0; i < amount; i++) {
			whiteSpace.append("\t");
		}

		// builds string representation
		StringBuffer stringBuffer = new StringBuffer();

		stringBuffer.append(whiteSpace);
		stringBuffer.append(String.format("%s: %s {\n",
				getClass().getSimpleName(), Variable.getName(variables(), ", ")));

		stringBuffer.append(whiteSpace);
		stringBuffer.append("\tneighbors = { ");

		for (AbstractNode neighbor : getNeighbors()) {
			stringBuffer.append("\"" + neighbor.getName() + "\" ");
		}

		stringBuffer.append("};\n");

		stringBuffer.append("focus: " + focus + "\n");

		if (potential() != null) {
			stringBuffer.append(potential());
			stringBuffer.append("\n");
		} else
			stringBuffer.append("potential: nil\n");

		stringBuffer.append(whiteSpace);
		stringBuffer.append("};\n");

		return stringBuffer.toString();
	}

}
