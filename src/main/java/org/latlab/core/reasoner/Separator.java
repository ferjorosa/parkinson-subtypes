package org.latlab.core.reasoner;

import org.latlab.core.graph.AbstractNode;
import org.latlab.core.util.DiscreteVariable;
import org.latlab.core.util.Function;
import org.latlab.core.util.Pair;
import org.latlab.core.util.Variable;

import java.util.*;

/**
 * Note: The potential has to be assigned before it can be used.
 * 
 * @author leonard
 * 
 */
public class Separator extends CliqueTreeNode {
	private DiscreteVariable variable;
	private Map<Clique, Message> messages;
	private Message lastMessage = null;
	private Function potential = null;

	public Separator(NaturalCliqueTree tree, DiscreteVariable variable) {
		super(tree, variable.getName());
		this.variable = variable;
	}

	@Override
	public void reset() {
		if (messages == null)
			messages = new HashMap<Clique, Message>(getNeighbors().size());

		// it does not clear the messages, since they should be either released
		// by the previous propagation or are intended to be kept

		potential = null;
	}

	public DiscreteVariable variable() {
		return variable;
	}

	/**
	 * Returns the potential of the separator. This potential is valid only
	 * after the distribution phrase and normalization of a propagation.
	 * Otherwise it returns {@code null}. This potential is maintained for
	 * finding the marginal probability of the variables contained.
	 * 
	 * TODO This potential may not be valid for local EM.
	 */
	@Override
	public Function potential() {
		return potential;
	}

	/**
	 * Stores the message coming from the given {@code clique}.
	 * 
	 * @param clique
	 *            from which the message is originated
	 * @param message
	 *            message from the originating clique
	 * 
	 */
	public void putMessage(Clique clique, Message message) {
		messages.put(clique, message);
		lastMessage = message;
	}

	public Message getMessage(Clique clique) {
		return messages.get(clique);
	}

	/**
	 * Sets the potential to the last message.
	 */
	public void setPotential() {
		if (lastMessage != null) {
			// the message is cloned to avoid affecting the stored messages
			potential = lastMessage.function.clone();
//			potential.normalize(constant);
		}
	}

	@Override
	public List<Variable> variables() {
		return Collections.singletonList((Variable) variable);
	}

	@Override
	public <T> T accept(Visitor<T> visitor) {
		return visitor.visit(this);
	}

	@Override
	public List<DiscreteVariable> discreteVariables() {
		return Collections.singletonList(variable);
	}

	/**
	 * Whether to release all messages, or only those messages in focus, so that
	 * they are computed again next time.
	 * 
	 * @param all
	 *            whether to release all messages
	 */
	protected void release(boolean all) {
		if (messages == null)
			return;

		if (all) {
			messages.clear();
		} else {
			Iterator<Clique> origin = messages.keySet().iterator();
			while (origin.hasNext()) {
				if (origin.next().focus()) {
					origin.remove();
				}
			}
		}
	}

	/**
	 * Returns whether this separator is within boundary of the focus subtree.
	 * It is meaningful only if the focus is set properly in the clique tree.
	 * 
	 * @return whether this separator is within focus boundary
	 */
	public boolean withinFocusBoundary() {
		for (AbstractNode node : getNeighbors()) {
			Clique clique = (Clique) node;
			if (clique.focus())
				return true;
		}

		return false;
	}

	/**
	 * Returns a memento instance for restoring later, or {@code null}
	 * if no messages is contained by this separator currently.
	 * 
	 * @return
	 */
	public MessageMemento createMessageMemento() {
		return new MessageMemento(messages);
	}

	public void setMessageMemento(MessageMemento memento) {
		if (memento == null || memento.isEmpty())
			return;

		if (messages == null) {
			messages = new HashMap<Clique, Message>();
		} else {
			messages.clear();
		}
		
		memento.putMessagesInto(messages);
	}

	/**
	 * For storing the messages in this separator, and restoring the messages
	 * later from this class.
	 * 
	 * @author leonard
	 * 
	 */
	public static class MessageMemento {
		private final List<Pair<Clique, Message>> messages;

		private MessageMemento(Map<Clique, Message> messages) {
			if (messages == null || messages.size() == 0) {
				this.messages = null;
			} else {
				this.messages = new ArrayList<Pair<Clique, Message>>(messages
						.size());
				for (Map.Entry<Clique, Message> entry : messages.entrySet()) {
					this.messages.add(new Pair<Clique, Message>(
							entry.getKey(), entry.getValue()));
				}
			}
		}

		/**
		 * Returns the map of messages from the key clique. It can be {@code
		 * null} if there is no such message.
		 * 
		 * @return map of messages or {@code null}
		 */
		private void putMessagesInto(Map<Clique, Message> map) {
			for (Pair<Clique, Message> pair : messages) {
				map.put(pair.first, pair.second);
			}
		}
		
		private boolean isEmpty() {
			return messages == null || messages.size() == 0;
		}
	}
}
