package org.latlab.core.io.bif;

import org.latlab.core.graph.AbstractNode;
import org.latlab.core.io.BeliefNodeProperties;
import org.latlab.core.io.BeliefNodeProperty;
import org.latlab.core.io.BeliefNodeProperty.ConnectionConstraint;
import org.latlab.core.io.Writer;
import org.latlab.core.learner.geast.IModelWithScore;
import org.latlab.core.model.*;
import org.latlab.core.util.*;

import java.awt.*;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

/**
 * Writes Bayesian networks in BIF format.
 * 
 * @author leonard
 * 
 */
public class BifWriter implements Writer {

    private final static String VARIABLE_NAME_FORMAT_STRING = "\"%s\"";

    /**
     * Constructs this writer with an underlying output stream, using the
     * default UTF-8 encoding.
     * 
     * @param output
     *            output stream where the network is written to.
     * @throws UnsupportedEncodingException
     */
    public BifWriter(OutputStream output) throws UnsupportedEncodingException {
        this(output, false, -1, "UTF-8");
    }

    /**
     * Constructs this writer with an underlying output stream, using the
     * default UTF-8 encoding.
     * 
     * @param output
     *            output stream where the network is written to.
     * @param useTableFormat
     *            whether to use table format in probability definition
     * @param precision
     *            number of decimal points in the float numbers in probability
     *            tables
     * @throws UnsupportedEncodingException
     */
    public BifWriter(OutputStream output, boolean useTableFormat, int precision)
        throws UnsupportedEncodingException {
        this(output, useTableFormat, precision, "UTF-8");
    }

    /**
     * Constructs this writer with an underlying output stream.
     * 
     * @param output
     *            output stream where the network is written to.
     * @param useTableFormat
     *            whether to use table format in probability definition
     * @param precision
     *            number of decimal points in the float numbers in probability
     *            tables
     * @param encoding
     *            charset used for the output.
     * @throws UnsupportedEncodingException
     */
    public BifWriter(
        OutputStream output, boolean useTableFormat, int precision,
        String encoding) throws UnsupportedEncodingException {
        this.useTableFormat = useTableFormat;
        writer = new PrintWriter(new OutputStreamWriter(output, encoding));
        this.precision = precision;
    }

    /**
     * Writes the network.
     */
    public void write(BayesNet network) {
        write(network, null);
    }

    public void write(BayesNet network, BeliefNodeProperties nodeProperties) {
        write(network, nodeProperties, null, true);
    }

    public void write(
        BayesNet network, BeliefNodeProperties nodeProperties, DataSet data) {
        write(network, nodeProperties, data, true);
    }

    public void write(
        BayesNet network, BeliefNodeProperties nodeProperties, DataSet data,
        boolean close) {
        writeNetworkDeclaration(network);
        writeVariables(network, nodeProperties);
        writeProbabilities(network);
        if (data != null)
            writeScore(network, data);

        if (close)
            writer.close();
    }

    /**
     * Writes the network and also its scores.
     * 
     * @param estimation
     *            holds the network and scores
     */
    public void write(IModelWithScore estimation) {
        write(estimation.model(), null, null, false);
        writeScore(estimation);
        writer.close();
    }

    /**
     * Writes the network declaration.
     * 
     * @param network
     *            network to write.
     */
    private void writeNetworkDeclaration(BayesNet network) {
        writer.format("network \"%s\" {\n}\n", network.getName());
        writer.println();
    }

    /**
     * Writes the variables part.
     * 
     * @param network
     *            network to write.
     */
    private void writeVariables(
        BayesNet network, BeliefNodeProperties nodeProperties) {
        LinkedList<AbstractNode> nodes = network.getNodes();
        for (AbstractNode node : nodes) {
            // get the node property
            final BeliefNodeProperty property =
                nodeProperties != null ? nodeProperties.get(node) : null;

            ((BeliefNode) node).accept(new BeliefNode.Visitor<Void>() {
                public Void visit(DiscreteBeliefNode node) {
                    writeDiscreteNode(node, property);
                    return null;
                }

                public Void visit(ContinuousBeliefNode node) {
                    writeContinuousNode(node, property);
                    return null;
                }
            });
        }
    }

    /**
     * Writes the information of a belief node.
     * 
     * @param node
     *            node to write.
     */
    private void writeDiscreteNode(
        DiscreteBeliefNode node, BeliefNodeProperty property) {
        List<String> states = node.getVariable().getStates();

        writer.format("variable \"%s\" {\n", node.getName());

        // write the states
        writer.format("\ttype discrete[%d] { ", states.size());
        for (String state : states) {
            writer.format("\"%s\" ", state);
        }
        writer.println("};");

        writeProperties(property);

        writer.println("}");
        writer.println();
    }

    /**
     * Writes the information of a belief node.
     * 
     * @param node
     *            node to write.
     */
    private void writeContinuousNode(
        ContinuousBeliefNode node, BeliefNodeProperty property) {

        // write the variables one by one
        for (SingularContinuousVariable variable : node.getVariable()
            .variables()) {
            writer.format("variable \"%s\" {\n", variable.getName());

            writer.println("\ttype continuous;");

            writeProperties(property);

            writer.println("}");
            writer.println();
        }
    }

    /**
     * Writes the properties for a node.
     * 
     * @param property
     *            property to write
     */
    private void writeProperties(BeliefNodeProperty property) {
        // write the position if necessary
        if (property != null) {
            Point point = property.getPosition();
            if (point != null) {
                writer.format(
                    "\tproperty \"position = (%d, %d)\";\n", point.x, point.y);
            }

            int angle = property.getRotation();
            if (angle != BeliefNodeProperty.DEFAULT_ROTATION) {
                writer.format("\tproperty \"rotation = %d\";\n", angle);
            }

            BeliefNodeProperty.FrameType frame = property.getFrame();
            if (frame != BeliefNodeProperty.DEFAULT_FRAME_TYPE) {
                if (frame == BeliefNodeProperty.FrameType.NONE) {
                    writer.write("\tproperty \"frame = none\";\n");
                } else if (frame == BeliefNodeProperty.FrameType.OVAL) {
                    writer.write("\tproperty \"frame = oval\";\n");
                } else if (frame == BeliefNodeProperty.FrameType.RECTANGLE) {
                    writer.write("\tproperty \"frame = rectangle\";\n");
                }
            }

            String label = property.getLabel();
            if (label != null) {
                writer.format("\tproperty \"label = '%s'\";\n", label);
            }

            Color color = property.getForeColor();
            if (color != null) {
                writer.format(
                    "\tproperty \"foreColor = (%d, %d, %d)\";\n", color
                        .getRed(), color.getGreen(), color.getBlue());
            }

            color = property.getBackColor();
            if (color != null) {
                writer.format(
                    "\tproperty \"backColor = (%d, %d, %d)\";\n", color
                        .getRed(), color.getGreen(), color.getBlue());
            }

            color = property.getLineColor();
            if (color != null) {
                writer.format(
                    "\tproperty \"lineColor = (%d, %d, %d)\";\n", color
                        .getRed(), color.getGreen(), color.getBlue());
            }

            String font = property.getFontName();
            if (font != null) {
                writer.format("\tproperty \"font = '%s'\";\n", font);
            }

            int fontSize = property.getFontSize();
            if (fontSize > 0) {
                writer.format("\tproperty \"fontSize = %d\";\n", fontSize);
            }

            ConnectionConstraint constraint =
                property.getConnectionConstraint();
            if (constraint != null) {
                writer.format(
                    "\tproperty \"connectionConstraint = '%s'\";\n", constraint
                        .name());
            }
        }

    }

    /**
     * Writes the probabilities definition part.
     * 
     * @param network
     *            network to write.
     */
    private void writeProbabilities(BayesNet network) {
        LinkedList<AbstractNode> nodes = network.getNodes();
        for (AbstractNode node : nodes) {
            writeProbabilities((BeliefNode) node);
        }
    }

    /**
     * Writes the probabilities definition for a belief node.
     * 
     * @param node
     *            node to write.
     */
    private void writeProbabilities(BeliefNode node) {
        final List<DiscreteVariable> parents =
            node.getDiscreteParentVariables();

        // write the related variables
        writer.print("probability (");
        writer.print(getNodeVariableNames(node));

        // check if it has parent variables
        if (parents.size() > 0) {
            writer.print(" | ");
            boolean first = true;
            for (DiscreteVariable parent : parents) {
                if (!first) {
                    writer.print(", ");
                }

                writer.format("\"%s\"", parent.getName());
                first = false;
            }
        }

        writer.println(") {");

        // write the probabilities table for different type of node
        node.accept(new BeliefNode.Visitor<Void>() {
            public Void visit(DiscreteBeliefNode node) {
                if (useTableFormat)
                    writeProbabilitiesTable(node.potential(), node
                        .getVariable(), parents);
                else
                    writeProbabilitiesWithStates(node.potential(), node
                        .getVariable(), parents);

                return null;
            }

            public Void visit(ContinuousBeliefNode node) {
                if (useTableFormat)
                    writeProbabilitiesTable(node.potential(), node
                        .getVariable(), parents);
                else
                    writeProbabilitiesWithStates(node.potential(), node
                        .getVariable(), parents);

                return null;
            }
        });

        writer.println("}");
        writer.println();
    }

    /**
     * Returns the string used for node variable names in the probability
     * definition. For example, [{@code "Y1", "Y2"}] or [{@code "X1"}].
     * 
     * @param node
     *            node of the probability definition
     * @return variable names
     */
    private String getNodeVariableNames(BeliefNode node) {
        return node.accept(new BeliefNode.Visitor<String>() {
            public String visit(ContinuousBeliefNode node) {
                return getVariableListName(node.getVariable().variables());
            }

            public String visit(DiscreteBeliefNode node) {
                return getVarialeName(node.getVariable());
            }
        });
    }

    private String getVariableListName(Collection<? extends Variable> variables) {
        StringBuilder builder = new StringBuilder();
        boolean first = true;
        for (Variable variable : variables) {
            if (!first)
                builder.append(", ");

            builder.append(getVarialeName(variable));
            first = false;
        }

        return builder.toString();
    }

    private String getVarialeName(Variable variable) {
        return String.format(VARIABLE_NAME_FORMAT_STRING, variable.getName());
    }

    private void writeProbabilitiesTable(
        Function function, DiscreteVariable variable,
        List<DiscreteVariable> parents) {
        List<DiscreteVariable> variables =
            new ArrayList<DiscreteVariable>(parents.size() + 1);
        variables.add(variable);
        variables.addAll(parents);

        double[] cells = function.getCells(variables);
        writer.print("\ttable ");
        for (int i = 0; i < cells.length; i++) {
            writeProbability(cells[i]);
            if (i != cells.length - 1) {
                writer.print(" ");
            }
        }
        writer.println(";");
    }

    private void writeProbabilitiesWithStates(
        Function function, DiscreteVariable variable,
        List<DiscreteVariable> parents) {
        // use table format for root variable
        if (parents.size() == 0) {
            writeProbabilitiesTable(function, variable, parents);
            return;
        }

        // put the parent variables at the beginning for iteration
        List<DiscreteVariable> order =
            new ArrayList<DiscreteVariable>(parents.size() + 1);
        order.addAll(parents);
        order.add(variable);

        FunctionIterator iterator = new FunctionIterator(function, order);
        iterator.iterate(new StateVisitor());
    }

    private void writeProbabilitiesTable(
        CGPotential potential, JointContinuousVariable variable,
        List<DiscreteVariable> parents) {
        boolean first = true;

        // this assumes a single parent variable
        writer.print("\ttable ");
        for (double entry : potential.getEntries(parents)) {
            if (!first)
                writer.print(" ");
            writeProbability(entry);
            first = false;
        }
        writer.println(";");
    }

    private void writeProbabilitiesWithStates(
        CGPotential potential, JointContinuousVariable variable,
        List<DiscreteVariable> parents) {

        // this assumes a single parent variable
        DiscreteVariable parent = parents.iterator().next();
        for (int i = 0; i < parent.getCardinality(); i++) {
            boolean first = true;
            writer.format("\t(\"%s\") ", parent.getStates().get(i));

            List<Double> entries = potential.get(i).getEntries();
            for (double entry : entries) {
                if (!first)
                    writer.print(" ");
                writeProbability(entry);
                first = false;
            }
            writer.println(";");
        }
    }

    private void writeScore(BayesNet network, DataSet data) {
        writer.println();
        writer.format("//Loglikelihood: %f\n", network.getLoglikelihood(data));
        writer.format("//BIC Score: %f\n", network.getBICScore(data));
        writer.println();
    }

    private void writeScore(IModelWithScore estimation) {
        writer.println();
        writer.format("// Loglikelihood: %f\n", estimation.loglikelihood());
        writer.format("// BIC Score: %f\n", estimation.BicScore());
        writer.println();
    }

    private void writeProbability(double value) {
        if (precision >= 0) {
            writer.format("%." + precision + "f", value);
        } else {
            writer.print(value);
        }
    }

    /**
     * The print writer encapsulating the underlying output stream.
     */
    private final PrintWriter writer;

    private boolean useTableFormat = false;
    private int precision = 2;

    private class StateVisitor implements FunctionIterator.Visitor {
        public void visit(
            List<DiscreteVariable> order, int[] states, double value) {
            // the node state and variable (instead of parent variables)
            int nodeState = states[states.length - 1];
            DiscreteVariable nodeVariable = order.get(states.length - 1);

            if (nodeState == 0) {
                writeStart(order, states);
            }

            writeProbability(value);

            if (nodeState == nodeVariable.getCardinality() - 1) {
                writeEnd();
            } else {
                writer.print(" ");
            }
        }

        private void writeStart(List<DiscreteVariable> order, int[] states) {
            writer.print("\t(");
            // write parent states, which excludes the last state
            for (int i = 0; i < states.length - 1; i++) {
                String stateName = order.get(i).getStates().get(states[i]);
                writer.format("\"%s\"", stateName);

                if (i < states.length - 2) {
                    writer.write(" ");
                }
            }

            writer.print(") ");
        }

        private void writeEnd() {
            writer.println(";");
        }
    }
}
