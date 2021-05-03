package org.latlab.core.reasoner;

import org.latlab.core.graph.AbstractNode;
import org.latlab.core.learner.geast.Focus;
import org.latlab.core.model.BeliefNode;
import org.latlab.core.model.CGPotential;
import org.latlab.core.model.Gltm;
import org.latlab.core.reasoner.Clique.NeighborVisitor;
import org.latlab.core.reasoner.Separator.MessageMemento;
import org.latlab.core.util.BuildConfig;
import org.latlab.core.util.DiscreteVariable;
import org.latlab.core.util.Function;
import org.latlab.core.util.SingularContinuousVariable;

import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.*;

/**
 * Propagation using a natural clique tree representation.
 * 
 * <p>
 * It supports setting propagation focus. At the end of the first time of
 * propagation, the messages stored at the separators are not removed. In the
 * subsequent propagations, messages are only passed around under the subtree
 * under focus, and the messages stored in the separators are used when the
 * cliques need to retrieve messages outside the focus.
 * 
 * <p>
 * This propagation under focus can be used when only parameters in one subtree
 * are being updated and the same evidence is being used during a number of
 * propagations. And in this case, the propagation should not be reused when the
 * parameters outside the focus have been modified.
 * 
 * <p>
 * The propagation needs to do normalization before the completion of message
 * passing. Otherwise the probability distribution will become unstable (all
 * goes to zero) and lose the difference between the different states.
 * 
 * @author leonard
 * 
 */
public class NaturalCliqueTreePropagation {

    // for profiling
    public static class ComputeTime {
        public final static int SIZE = 6;

        public enum Types {
            INITIALIZATION, CONTINUOUS_EVIDENCE, DISCRETE_EVIDENCE, MESSAGES, NORMALIZATION, OVERALL
        }

        public ComputeTime() {
            if (BuildConfig.PROFILE) {
                values = new long[Types.values().length];
                bean = ManagementFactory.getThreadMXBean();
            } else {
                values = null;
                bean = null;
            }
        }

        public final long[] values;
        private final ThreadMXBean bean;

        public static void start(ComputeTime time, Types type) {
            if (BuildConfig.PROFILE)
                time.values[type.ordinal()] =
                        time.bean.getCurrentThreadCpuTime();
            // if (BuildConfig.PROFILE)
            // time.values[type.ordinal()] =
            // time.bean.getCurrentThreadUserTime();
        }

        public static void stop(ComputeTime time, Types type) {
            if (BuildConfig.PROFILE) {
                time.values[type.ordinal()] =
                        time.bean.getCurrentThreadCpuTime()
                                - time.values[type.ordinal()];
                // time.values[type.ordinal()] =
                // time.bean.getCurrentThreadUserTime()
                // - time.values[type.ordinal()];
            }
        }

        public static String format(ComputeTime time) {
            if (BuildConfig.PROFILE) {
                return String.format("%d,%d,%d,%d,%d,%d", time.values[0],
                        time.values[1], time.values[2], time.values[3],
                        time.values[4], time.values[5]);
            } else
                return null;
        }

        public void add(ComputeTime t) {
            for (int i = 0; i < Types.values().length; i++) {
                values[i] += t.values[i];
            }
        }
    }

    public ComputeTime computeTime =
            BuildConfig.PROFILE ? new ComputeTime() : null;

    private Evidences evidences;

    private final NaturalCliqueTree tree;

    private Gltm model;

    private double loglikelihood = 0;

    private final boolean focusSpecified;

    /**
     * Whether the evidences have been absorbed. Relevant only when the focus
     * has been specified. Otherwise it should remain false.
     */
    private boolean evidencesAbsorbed = false;

    private int messagesPassed = 0;

    /**
     * @param focusSpecified
     */
    private NaturalCliqueTreePropagation(Gltm structure,
            boolean focusSpecified) {
        // it is assumed that the structure contains the valid parameters unless
        // another model is given
        this.model = structure;

        tree = new NaturalCliqueTree(structure);
        evidences = new Evidences();
        this.focusSpecified = focusSpecified;
    }

    public NaturalCliqueTreePropagation(Gltm model) {
        this(model, false);
    }

    public NaturalCliqueTreePropagation(Gltm model, Focus focus) {
        this(model, true);
        assert focus != null;

        tree.setFocus(focus);
    }

    /**
     * Returns the clique tree used by this propagation.
     * 
     * @return clique tree used by this propagation.
     */
    public NaturalCliqueTree cliqueTree() {
        return tree;
    }

    /**
     * Sets the model from which the parameters are initialized from. The given
     * model should have the same structure (containing the same set of
     * variables as well) of the model given to the constructor.
     * 
     * @param model
     *            from which parameters are initialized from
     */
    public void useModel(Gltm model) {
        this.model = model;
    }

    /**
     * Performs the propagation and returns the likelihood, which is
     * P(Evidences).
     * 
     * @return likelihood
     */
    public void propagate() {
        messagesPassed = 0;

        ComputeTime.start(computeTime, ComputeTime.Types.OVERALL);

        // initialization

        ComputeTime.start(computeTime, ComputeTime.Types.INITIALIZATION);
        initializePotentials();
        ComputeTime.stop(computeTime, ComputeTime.Types.INITIALIZATION);

        ComputeTime.start(computeTime, ComputeTime.Types.DISCRETE_EVIDENCE);
        absorbDiscreteEvidence();
        ComputeTime.stop(computeTime, ComputeTime.Types.DISCRETE_EVIDENCE);

        ComputeTime.start(computeTime, ComputeTime.Types.CONTINUOUS_EVIDENCE);
        absorbContinuousEvidence();
        ComputeTime.stop(computeTime, ComputeTime.Types.CONTINUOUS_EVIDENCE);

        if (focusSpecified)
            evidencesAbsorbed = true;

        // propagation
        ComputeTime.start(computeTime, ComputeTime.Types.MESSAGES);
        collectMessage(tree.pivot());
        distributeMessage(tree.pivot());
        ComputeTime.stop(computeTime, ComputeTime.Types.MESSAGES);

        ComputeTime.start(computeTime, ComputeTime.Types.NORMALIZATION);
        // keep the potential of the pivot under proper normalization
        tree.pivot().normalize(Double.NaN);

        loglikelihood = tree.pivot().logNormalization();
        ComputeTime.stop(computeTime, ComputeTime.Types.NORMALIZATION);

        // likelihood =
        // tree.pivot().normalization().multiply(
        // new BigDecimal(Math.exp(log), MathContext.DECIMAL64),
        // MathContext.DECIMAL64);

        setSeparatorPotentials();

        release();

        // // 1e-309 is the (empirical) smallest number that doesn't lead to NaN
        // // after division
        // if (likelihood <= 1e-309 || Double.isNaN(likelihood)) {
        // throw new ImpossibleEvidenceException(model.clone(),
        // evidences.copy());
        // }

        ComputeTime.stop(computeTime, ComputeTime.Types.OVERALL);

        // 1e-309 is the (empirical) smallest number that doesn't lead to NaN
        // after division
         if (loglikelihood == Double.NEGATIVE_INFINITY
                 || Double.isNaN(loglikelihood)) {
             throw new ImpossibleEvidenceException(model.clone(),
                     evidences.copy());
         }
    }

    /**
     * Initializes the potentials in the clique tree.
     */
    void initializePotentials() {
        for (AbstractNode node : tree.getNodes()) {
            ((CliqueTreeNode) node).reset();
        }

        // TODO LP: initialize the cliques outside focus only for the first time

        for (AbstractNode node : model.getNodes()) {
            BeliefNode beliefNode = (BeliefNode) node;
            Clique clique = tree.getClique(beliefNode.getVariable());
            clique.combine(beliefNode.potential());
        }
    }

    /**
     * Returns the natural log of the product of the constants multiplied to the
     * parameters when the evidence is absorbed.
     * 
     * @return
     */
    void absorbContinuousEvidence() {

        for (Map.Entry<SingularContinuousVariable, Double> entry : evidences.continuous().entrySet()) {
            SingularContinuousVariable variable = entry.getKey();

            MixedClique clique = tree.getClique(variable);

            // ignore variable not contained in this model
            if (clique == null) {
                continue;
            }

            if (evidencesAbsorbed && !clique.focus())
                continue;

            // proceeds if evidences haven't been absorbed or the clique is
            // under focus

            clique.absorbEvidence(variable, entry.getValue());
        }
    }

    void absorbDiscreteEvidence() {
        for (Map.Entry<DiscreteVariable, Integer> entry : evidences.discrete().entrySet()) {
            final DiscreteVariable variable = entry.getKey();
            final int state = entry.getValue();

            Clique clique = tree.getClique(variable);

            // ignore variable not contained in this model
            if (clique == null) {
                continue;
            }

            if (evidencesAbsorbed && !clique.focus())
                continue;

            // proceeds if evidences haven't been absorbed or the clique is
            // under focus

            // set the evidence on the cliques containing the variable
            clique.potential().timesIndicator(variable, state);

            // set the evidence on the neighboring cliques if they also contain
            // the evidence variable
            clique.visitNeighbors(new NeighborVisitor(null) {
                @Override
                public void visit(Separator separator, Clique neighbor) {
                    if (!neighbor.contains(variable))
                        return;

                    neighbor.potential().timesIndicator(variable, state);
                }
            });
        }
    }

    private void collectMessage(final Clique sink) {
        sink.visitNeighbors(new NeighborVisitor(null) {
            @Override
            public void visit(Separator separator, Clique neighbor) {
                collectMessage(sink, separator, neighbor);
            }
        });
    }

    private void collectMessage(final Clique sink, final Separator separator,
            final Clique source) {
        // if the separator has already cached a message from the source, then
        // don't need to ask the source to collect messages from its neighbors
        if (separator.getMessage(source) == null) {
            source.visitNeighbors(new NeighborVisitor(separator) {
                @Override
                public void visit(Separator separator1, Clique neighbor) {
                    collectMessage(source, separator1, neighbor);
                }
            });
        }

        sendMessage(source, separator, sink, false);
    }

    private void distributeMessage(final Clique source) {
        source.visitNeighbors(new NeighborVisitor(null) {
            @Override
            public void visit(Separator separator, Clique neighbor) {
                distributeMessage(source, separator, neighbor);
            }
        });
    }

    private void distributeMessage(final Clique source,
            final Separator separator, final Clique sink) {
        // if the sink is not in the focus, don't need to distribute to it.
        if (!sink.focus())
            return;

        sendMessage(source, separator, sink, true);

        sink.visitNeighbors(new NeighborVisitor(separator) {
            @Override
            public void visit(Separator separator1, Clique neighbor) {
                distributeMessage(sink, separator1, neighbor);
            }
        });
    }

    /**
     * Sends a message from {@code source} to {@code sink} through the {@code
     * separator} between them.
     * 
     * @param source
     * @param separator
     * @param sink
     * @param distributing
     *            whether it is in the distribution phrase
     */
    private void sendMessage(Clique source, Separator separator, Clique sink,
            boolean distributing) {

        Message sourceMessage = separator.getMessage(source);
        if (sourceMessage == null) {
            sourceMessage = source.computeMessage(separator);
            separator.putMessage(source, sourceMessage);
        }

        if (distributing) {
            Message sinkMessage = separator.getMessage(sink);
            assert sinkMessage != null;
            if (sinkMessage != null) {
                // since the source message is retrieved from the map, changing
                // it would affect subsequent retrieval, so we need to operate
                // on a clone instead
                sourceMessage = sourceMessage.clone();
                sourceMessage.divide(sinkMessage);
            }
        }

        sink.combine(sourceMessage);

        messagesPassed++;
    }

    // private double normalize() {
    // tree.pivot().normalize(Double.NaN);
    // likelihood = tree.pivot().normalization().doubleValue();
    //
    // // for (AbstractNode node : tree.getNodes()) {
    // // // the root has been normalized
    // // if (node != tree.pivot())
    // // ((CliqueTreeNode) node).normalize(likelihood);
    // // }
    //
    // return likelihood;
    //
    // // Since the normalization is carried out at each step of message
    // // combination, it doesn't need to do it again. The normalization
    // // constant used by the pivot is the likelihood.
    // // return tree.pivot().normalization;
    // }

    private void setSeparatorPotentials() {
        for (Separator s : tree.separators()) {
            s.setPotential();
        }
    }

    /**
     * Releases the stored messages in the separators.
     * 
     * @param force
     *            whether to force the release of messages regardless of whether
     *            it is doing local propagation
     */
    private void release(boolean force) {
        // the messages are kept at the separators when the propagation focus
        // has been specified, so that the messages outside the focus subtree
        // are not computed in the future propagation
        for (Separator separator : tree.separators()) {
            separator.release(force || !focusSpecified);
        }
    }

    private void release() {
        release(false);
    }

    public Function getMarginal(DiscreteVariable variable) {
        return tree.getClique(variable).potential().marginalize(variable);
    }

    public CGPotential getMarginal(SingularContinuousVariable variable) {
        return tree.getClique(variable).potential().marginalize(variable);
    }

    public Function getMarginal(Collection<DiscreteVariable> variables) {
        return getMarginal(new HashSet<DiscreteVariable>(variables));
    }

    public Function getMarginal(Set<DiscreteVariable> variables) {
        if (variables.size() < 2) {
            return getMarginal(variables.iterator().next());
        }

        Set<CliqueTreeNode> subtree = tree.findMinimalSubtree(variables);

        // find the first clique. If the model has only one discrete node,
        // the clique tree contains only one mixed clique and so a discrete
        // pivot can't be found. However, since it has assumed that there are
        // more than 2 discrete variables, this case won't happen.
        DiscreteClique pivot = null;
        for (CliqueTreeNode node : subtree) {
            if (node instanceof DiscreteClique) {
                pivot = (DiscreteClique) node;
                break;
            }
        }

        assert pivot != null;

        List<Message> messages =
                collectMessagesFromNeighbors(subtree, variables, pivot, null);

        Function product = pivot.potential();
        if (messages.size() > 0) {
            product = product.times(Message.computeProduct(messages).function);
        }

        if (product.getDimension() != variables.size()) {
            product = product.marginalize(variables);
        }

        return product;
    }

    /**
     * Computes the joint function of the variables by combining from the
     * potential of the source clique node.
     * 
     * @param separator
     * @param source
     * @param variables
     * @return
     */
    private Message computeJointBySendingMessage(
            final Set<CliqueTreeNode> subtree,
            final Set<DiscreteVariable> variables, final Clique source,
            final Separator separator) {

        List<Message> messages = collectMessagesFromNeighbors(subtree,
                variables, source, separator);
        Message product = Message.computeProduct(messages);

        Message message = source.computeMessage(product, separator, variables);
        message.divide(separator.potential());
        return message;
    }

    private List<Message> collectMessagesFromNeighbors(
            final Set<CliqueTreeNode> subtree,
            final Set<DiscreteVariable> variables, final Clique source,
            final Separator origin) {

        final List<Message> messages = origin == null ? new ArrayList<Message>()
                : new ArrayList<Message>(origin.getDegree() - 1);

        // collect messages from neighbors
        source.visitNeighbors(new NeighborVisitor(origin) {

            @Override
            public void visit(Separator separator, Clique neighbor) {
                if (!subtree.contains(neighbor))
                    return;

                Message message = computeJointBySendingMessage(subtree,
                        variables, neighbor, separator);
                messages.add(message);
            }
        });

        return messages;
    }

    public Gltm model() {
        return model;
    }

    public Evidences evidences() {
        return evidences;
    }

    /**
     * Uses the given evidences. If it is {@code null}, it uses a new instance
     * of {@code Evidences} created by itself.
     * 
     * @param evidences
     *            evidences to use
     */
    public void use(Evidences evidences) {
        if (evidences == null) {
            evidences = new Evidences();
        }

        this.evidences = evidences;
    }

    public double likelihood() {
        return Math.exp(loglikelihood);
    }

    /**
     * @return natural log of the likelihood
     */
    public double loglikelihood() {
        // return Math.log(likelihood.unscaledValue().doubleValue()) - LOG_10
        // * likelihood.scale();
        // return Math.log(likelihood);
        return loglikelihood;
    }

    /**
     * Returns the number of messages passed in the last propagation. Used for
     * debugging.
     * 
     * @return number of messages passed
     */
    public int messagesPassed() {
        return messagesPassed;
    }

    /**
     * Resets the internal states of this propagation after an local
     * propagation. It is not necessary to call this method when the focus is
     * not specified.
     */
    public void resetLocalPropagation() {
        evidencesAbsorbed = false;
        release(true);
    }

    /**
     * Clears the messages stored in the separators that are not within the
     * boundary of the focus subtree.
     */
    public void releaseSeparatorMessagesOutsideFocus() {
        for (Separator separator : tree.separators()) {
            if (!separator.withinFocusBoundary()) {
                separator.release(true);
            }
        }
    }

    /**
     * For storing some necessary states after a propagation made on a model on
     * focus subtree. The states can be recovered later so that it does not need
     * to propagate outside the focus subtree again. The states are valid even
     * if it has performed inference using some other evidences.
     * 
     * <p>
     * However, the states must match with the model structure and evidences
     * used when it is created. Moreover, the parameters of the model must be
     * the same outside the focus subtree.
     * 
     * 
     * @author leonard
     * 
     */
    public static class LocalPropagationMemento {
        private final List<MessageMemento> messages;

        private LocalPropagationMemento(List<MessageMemento> messages) {
            this.messages = messages;
        }
    }

    /**
     * Saves the states after a propagation on the focus subtree.
     * 
     * @return states for restoring later
     */
    public LocalPropagationMemento createLocalPropagationMemento() {
        // store the messages in the separator nodes if that separator is
        List<MessageMemento> messages =
                new ArrayList<MessageMemento>(tree.separators().size());

        for (Separator separator : tree.separators()) {
            if (separator.withinFocusBoundary()) {
                messages.add(separator.createMessageMemento());
            } else {
                // need to add a null value as a place holder
                messages.add(null);
            }
        }

        return new LocalPropagationMemento(messages);
    }

    /**
     * Restores the states of the propagation on particular model and evidences.
     * 
     * @param memento
     *            holds the states
     */
    public void setLocalPropagationMemento(LocalPropagationMemento memento) {
        // this assumes that the separator has a fixed order
        Iterator<MessageMemento> messageIterator = memento.messages.iterator();

        for (Separator separator : tree.separators()) {
            MessageMemento message = messageIterator.next();
            if (message != null) {
                separator.setMessageMemento(message);
            }
        }
    }
}
