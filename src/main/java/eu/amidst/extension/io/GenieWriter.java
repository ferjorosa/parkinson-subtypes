package eu.amidst.extension.io;

import eu.amidst.core.distribution.ConditionalLinearGaussian;
import eu.amidst.core.distribution.Normal;
import eu.amidst.core.distribution.Normal_MultinomialNormalParents;
import eu.amidst.core.distribution.Normal_MultinomialParents;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.utils.MultinomialIndex;
import eu.amidst.core.variables.Assignment;
import eu.amidst.core.variables.HashMapAssignment;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.stateSpaceTypes.RealStateSpace;
import eu.amidst.extension.util.GraphUtilsAmidst;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import java.io.File;
import java.util.List;
import java.util.stream.Collectors;

public class GenieWriter {

    private int numberOfDiscretizationIntervals;

    public GenieWriter() {
        this.numberOfDiscretizationIntervals = 10;
    }

    public GenieWriter(int numberOfDiscretizationIntervals) {
        this.numberOfDiscretizationIntervals = numberOfDiscretizationIntervals;
    }

    public void write(BayesianNetwork network, File file) throws ParserConfigurationException, TransformerException {
        DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
        DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
        Document doc = dBuilder.newDocument();

        doc.setXmlStandalone(false);

        Element rootElement = writeRoot(network, doc);
        writeNodes(network, doc, rootElement);
        writeExtensions(network, doc, rootElement);

        DOMSource source = new DOMSource(doc);

        TransformerFactory transformerFactory = TransformerFactory.newInstance();
        Transformer transformer = transformerFactory.newTransformer();
        StreamResult result = new StreamResult(file);
        transformer.setOutputProperty(OutputKeys.ENCODING, "ISO-8859-1");
        transformer.setOutputProperty(OutputKeys.INDENT, "yes");
        transformer.setOutputProperty("{http://xml.apache.org/xslt}indent-amount", "2");
        transformer.transform(source, result);
    }

    private Element writeRoot(BayesianNetwork network,
                              Document doc){
        Element rootElement = doc.createElement("smile");
        doc.appendChild(rootElement);

        /* Attributes */
        rootElement.setAttribute("version", "1.0");
        rootElement.setAttribute("id", network.getName());
        rootElement.setAttribute("numsamples", "10000");
        rootElement.setAttribute("discsamples", "10000");

        return rootElement;
    }

    private void writeNodes(BayesianNetwork network,
                            Document doc,
                            Element rootElement) {

        Element nodesElement = doc.createElement("nodes");
        rootElement.appendChild(nodesElement);

        List<Variable> sortedVars = GraphUtilsAmidst.topologicalSort(network.getDAG());

        for(Variable var: sortedVars)
            writeNode(var, network, doc, nodesElement);
    }

    private void writeNode(Variable variable,
                           BayesianNetwork network,
                           Document doc,
                           Element nodesElement) {

        if(variable.isDiscrete())
            writeDiscreteNode(variable, network, doc, nodesElement);

        else if(variable.isContinuous())
            writeContinuousNode(variable, network, doc, nodesElement);
    }

    private void writeDiscreteNode(Variable variable,
                                   BayesianNetwork network,
                                   Document doc,
                                   Element nodesElement) {

        /* Creamos el elemento correspondiente a la variable discreta */
        Element cptElement = doc.createElement("cpt");
        cptElement.setAttribute("id", variable.getName());
        nodesElement.appendChild(cptElement);

        /* 1 - Definimos sus estados */
        for (int i = 0; i < variable.getNumberOfStates(); i++) {
            Element stateElement = doc.createElement("state");
            stateElement.setAttribute("id", variable.getStateSpaceType().stringValue(i));
            cptElement.appendChild(stateElement);
        }

        /* 2 - Definimos sus padres */
        List<Variable> parentVariables = network.getDAG().getParentSet(variable).getParents();
        Element parentsElement = doc.createElement("parents");
        for (Variable parent : parentVariables) {
            parentsElement.appendChild(doc.createTextNode(parent.getName() + " "));
            cptElement.appendChild(parentsElement);
        }

        /* 3 - Definimos sus probabilidades */
        double[] probabilities = network.getConditionalDistribution(variable).getParameters();
        StringBuilder probabilitiesString = new StringBuilder();
        for (int i = 0; i < probabilities.length; i++) {
            probabilitiesString.append(probabilities[i]);
            probabilitiesString.append(" ");
        }
        Element probabilitiesElement = doc.createElement("probabilities");
        probabilitiesElement.appendChild(doc.createTextNode(probabilitiesString.toString()));
        cptElement.appendChild(probabilitiesElement);
    }

    private void writeContinuousNode(Variable variable,
                                     BayesianNetwork network,
                                     Document doc,
                                     Element nodesElement) {

        /* Creamos el elemento correspondiente a la variable continua */
        Element equationElement = doc.createElement("equation");
        equationElement.setAttribute("id", variable.getName());
        nodesElement.appendChild(equationElement);

        RealStateSpace stateSpace = variable.getStateSpaceType();

        /* 1 - Definimos sus padres */
        List<Variable> parentVariables = network.getDAG().getParentSet(variable).getParents();
        if(parentVariables.size() > 0 ) {
            Element parentsElement = doc.createElement("parents");
            for (Variable parent : parentVariables) {
                parentsElement.appendChild(doc.createTextNode(parent.getName() + " "));
                equationElement.appendChild(parentsElement);
            }
        }

        /* 2 - Definimos sus parametros */
        Element definitionElement = doc.createElement("definition");
        if (stateSpace.getMinInterval() == Double.NEGATIVE_INFINITY)
            throw new IllegalArgumentException("Attribute max and min values need to be specified. It can be done with DataUtils.defineAttributesMaxMinValues()");
        else
            definitionElement.setAttribute("lower", stateSpace.getMinInterval() + "");

        if (stateSpace.getMaxInterval() == Double.POSITIVE_INFINITY)
            throw new IllegalArgumentException("Attribute max and min values need to be specified. It can be done with DataUtils.defineAttributesMaxMinValues()");
        else
            definitionElement.setAttribute("upper", stateSpace.getMaxInterval() + "");

        StringBuilder definitionString = new StringBuilder();

        if(network.getConditionalDistribution(variable) instanceof Normal)
            writeNormal(variable, network, definitionString);
        else if(network.getConditionalDistribution(variable) instanceof ConditionalLinearGaussian)
            writeCLG(variable, network, parentVariables, definitionString);
        else if(network.getConditionalDistribution(variable) instanceof Normal_MultinomialParents)
            writeNormal_MultinomialParents(variable, network, parentVariables, definitionString);
        else if(network.getConditionalDistribution(variable) instanceof Normal_MultinomialNormalParents)
            writeNormal_MultinomialNormalParents(variable, network, parentVariables, definitionString);

        definitionElement.appendChild(doc.createTextNode(definitionString.toString()));
        equationElement.appendChild(definitionElement);

        /* 3 - Definimos su discretización */
        double min = stateSpace.getMinInterval();
        double max = stateSpace.getMaxInterval();
        double[] discretizationCutPoints = estimateDiscretizationCutPoints(min, max, this.numberOfDiscretizationIntervals);

        Element discretizationElement =  doc.createElement("discretization");
        for(int i = 0; i < discretizationCutPoints.length; i++){
            Element intervalElement = doc.createElement("interval");
            intervalElement.setAttribute("upper", discretizationCutPoints[i] + "");
            discretizationElement.appendChild(intervalElement);
        }

        equationElement.appendChild(discretizationElement);
    }

    private static void writeNormal(Variable variable,
                                    BayesianNetwork network,
                                    StringBuilder definitionString) {

        Normal condDist = network.getConditionalDistribution(variable);
        definitionString.append(variable.getName());
        definitionString.append("=Normal(" + condDist.getMean() + "," + condDist.getVariance() + ")");
    }

    private static void writeCLG(Variable variable,
                                 BayesianNetwork network,
                                 List<Variable> parentVariables,
                                 StringBuilder definitionString) {

        ConditionalLinearGaussian condDist = network.getConditionalDistribution(variable);;
        definitionString.append(variable.getName() + "=");
        for(int i = 0; i < parentVariables.size(); i++) {
            double coef = condDist.getCoeffParents()[i];
            definitionString.append(coef + "*" + parentVariables.get(i).getName() + "+");
        }
        definitionString.append("Normal(" + condDist.getIntercept() + "," + condDist.getVariance() + ")");
    }

    private static void writeNormal_MultinomialParents(Variable variable,
                                                       BayesianNetwork network,
                                                       List<Variable> parentVariables,
                                                       StringBuilder definitionString) {

        Normal_MultinomialParents condDist = network.getConditionalDistribution(variable);
        definitionString.append(variable.getName() + "=");

        // If there is only one (discrete) parent
        if(parentVariables.size() == 1) {
            Variable parentVar = parentVariables.get(0);
            definitionString.append("Choose(");
            definitionString.append(parentVar.getName());

            HashMapAssignment parentAssignment = new HashMapAssignment();
            for (int i = 0; i < parentVar.getNumberOfStates(); i++) {
                parentAssignment.setValue(parentVar, i);
                Normal normal = condDist.getNormal(parentAssignment);
                definitionString.append(",Normal(" + normal.getMean() + "," + normal.getVariance() + ")");
            }
            definitionString.append(")");
        }
        // Multiple (discrete) parents
        else {
            for(int i = 0; i<condDist.getNumberOfParentAssignments();i++){
                Normal normal = condDist.getNormal(i);
                Assignment parentAssignment = MultinomialIndex.getVariableAssignmentFromIndex(condDist.getConditioningVariables(), i);

                // For the majority of cases we add an "If(And()..." clause
                if(i != (condDist.getNumberOfParentAssignments() - 1)){
                    definitionString.append("If(And(");
                    int parentCount = 0; // Count the parents because at the last one, the definition will be different
                    for(Variable parent: parentAssignment.getVariables()) {
                        parentCount++;
                        String parentState = parent.getStateSpaceType().stringValue(parentAssignment.getValue(parent));
                        if (parentCount != parentAssignment.getVariables().size())
                            definitionString.append(parent.getName() + "=\"" + parentState + "\",");
                        else
                            definitionString.append(parent.getName() + "=\"" + parentState + "\")");
                    }
                    definitionString.append(",Normal(" + normal.getMean() + "," + normal.getVariance() + "),");
                }
                // The last one is written as an "Else/Default" case, thus no need to add the "If(And()..." clause
                // In addition, it closes previous If(And()... parenthesis
                else {
                    definitionString.append("Normal(" + normal.getMean() + "," + normal.getVariance() + ")");
                    for(int parenthesisCount = 0; parenthesisCount < condDist.getNumberOfParentAssignments() - 1; parenthesisCount++)
                        definitionString.append(")");
                }
            }
        }
    }

    private static void writeNormal_MultinomialNormalParents(Variable variable,
                                                             BayesianNetwork network,
                                                             List<Variable> parentVariables,
                                                             StringBuilder definitionString) {

        Normal_MultinomialNormalParents condDist = network.getConditionalDistribution(variable);
        definitionString.append(variable.getName() + "=");
        List<Variable> discreteParentVariables = parentVariables.stream().filter(x->x.isDiscrete()).collect(Collectors.toList());
        List<Variable> continuousParentVariables = parentVariables.stream().filter(x->x.isContinuous()).collect(Collectors.toList());

        // If there is only one discrete parent and at least one continuous parent
        if(discreteParentVariables.size() == 1) {
            Variable parentVar = discreteParentVariables.get(0);
            definitionString.append("Choose(");
            definitionString.append(parentVar.getName());

            HashMapAssignment parentAssignment = new HashMapAssignment();
            for (int discState = 0; discState < parentVar.getNumberOfStates(); discState++) {
                definitionString.append(",");
                parentAssignment.setValue(parentVar, discState);
                ConditionalLinearGaussian clg = condDist.getNormal_NormalParentsDistribution(parentAssignment);
                for(int contParentIndex = 0; contParentIndex < continuousParentVariables.size(); contParentIndex++) {
                    double coef = clg.getCoeffParents()[contParentIndex];
                    definitionString.append(coef + "*" + parentVariables.get(contParentIndex).getName() + "+");
                }
                definitionString.append("Normal(" + clg.getIntercept() + "," + clg.getVariance() + ")");
            }
            definitionString.append(")");
        }
        // If there are multiple discrete parents and at least one continuous parent
        else {
            for(int i = 0; i < condDist.getNumberOfParentAssignments(); i++){
                ConditionalLinearGaussian clg = condDist.getNormal_NormalParentsDistribution(i);
                Assignment parentAssignment = MultinomialIndex.getVariableAssignmentFromIndex(discreteParentVariables, i); // TODO: discreteVars or conditioningVars?

                // For the majority of cases we add an "If(And()..." clause
                if(i != (condDist.getNumberOfParentAssignments() - 1)){
                    definitionString.append("If(And(");
                    int parentCount = 0; // Count the parents because at the last one, the definition will be different
                    for(Variable parent: parentAssignment.getVariables()) {
                        parentCount++;
                        String parentState = parent.getStateSpaceType().stringValue(parentAssignment.getValue(parent));
                        if (parentCount != parentAssignment.getVariables().size())
                            definitionString.append(parent.getName() + "=\"" + parentState + "\",");
                        else
                            definitionString.append(parent.getName() + "=\"" + parentState + "\")");
                    }
                    for(int clgIndex = 0; clgIndex < continuousParentVariables.size(); clgIndex++) {
                        double coef = clg.getCoeffParents()[clgIndex];
                        definitionString.append(coef + "*" + parentVariables.get(i).getName() + "+");
                    }
                    definitionString.append("Normal(" + clg.getIntercept() + "," + clg.getVariance() + "),");
                }
                // The last one is written as an "Else/Default" case, thus no need to add the "If(And()..." clause
                // In addition, it closes previous If(And()... parenthesis
                else {
                    for(int clgIndex = 0; clgIndex < continuousParentVariables.size(); clgIndex++) {
                        double coef = clg.getCoeffParents()[clgIndex];
                        definitionString.append(coef + "*" + parentVariables.get(i).getName() + "+");
                    }
                    definitionString.append("Normal(" + clg.getIntercept() + "," + clg.getVariance() + ")");
                    for(int parenthesisCount = 0; parenthesisCount < condDist.getNumberOfParentAssignments() - 1; parenthesisCount++)
                        definitionString.append(")");
                }
            }
        }
    }

    private void writeExtensions(BayesianNetwork network,
                                 Document doc,
                                 Element rootElement) {

        Element extensionsElement = doc.createElement("extensions");
        rootElement.appendChild(extensionsElement);

        Element genieElement = doc.createElement("genie");
        genieElement.setAttribute("version", "1.0");
        genieElement.setAttribute("app", "GeNIe 2.3.3705.0 ACADEMIC");
        genieElement.setAttribute("name", network.getName());
        genieElement.setAttribute("faultnameformat", "nodestate");
        extensionsElement.appendChild(genieElement);

        for (Variable var : network.getDAG().getVariables()) {

            Element nodeElement = doc.createElement("node");
            nodeElement.setAttribute("id", var.getName());
            genieElement.appendChild(nodeElement);

            Element nameElement = doc.createElement("name");
            nameElement.appendChild(doc.createTextNode(var.getName()));
            nodeElement.appendChild(nameElement);

            Element interiorElement = doc.createElement("interior");
            interiorElement.setAttribute("color", "e5f6f7");
            nodeElement.appendChild(interiorElement);

            Element outlineElement = doc.createElement("outline");
            outlineElement.setAttribute("color", "000080");
            nodeElement.appendChild(outlineElement);

            Element fontElement = doc.createElement("font");
            fontElement.setAttribute("color", "000080");
            fontElement.setAttribute("name", "Arial");
            fontElement.setAttribute("size", "10");
            fontElement.setAttribute("bold", "true");
            nodeElement.appendChild(fontElement);

            Element positionElement = doc.createElement("position");
            positionElement.appendChild(doc.createTextNode("100 100 100 100"));
            nodeElement.appendChild(positionElement);

            Element barchartElement = doc.createElement("barchart");
            barchartElement.setAttribute("active", "true");
            barchartElement.setAttribute("width", "160");
            barchartElement.setAttribute("height", "110");
            nodeElement.appendChild(barchartElement);

        }
    }

    private double[] estimateDiscretizationCutPoints(double min, double max, int n) {
        double[] cutPoints = new double[n];
        double dist = (max-min) / (double) n;
        double current = min;
        for(int i = 0; i < n; i++) {
            current = current + dist;
            cutPoints[i] = current;
        }

        return cutPoints;
    }
}
