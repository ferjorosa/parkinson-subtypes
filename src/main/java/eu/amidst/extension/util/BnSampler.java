package eu.amidst.extension.util;

import eu.amidst.core.datastream.*;
import eu.amidst.core.distribution.ConditionalDistribution;
import eu.amidst.core.distribution.UnivariateDistribution;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.variables.Assignment;
import eu.amidst.core.variables.HashMapAssignment;
import eu.amidst.core.variables.Variable;
import eu.amidst.extension.data.DataInstanceFromRawData;

import java.util.*;

/**
 * Forward sampling. It is a simple implementation designed to generate synthetic dataSets from Bayesian networks.
 * It could be easily made more efficient.
 */
/*
    Basicamente tenemos un doble for, uno que itera sobre N y otro que itera en orden por la lista de variables
    en orden topologico.

    - Por cada n in N, creamos un Map donde vamos a ir almacenando los valores de las variables visitadas
    - Ordenamos las variables segun su orden topologico en la red
    - Vamos sampleando en orden.
        * Si la variable no tiene padres la transformamos en univariante y sampleamos de ella
        * Si la variable tiene padres, buscamos en el map sus valores y creamos un assignment para obtener
          la distribucion univariante. Una vez la tenemos, sampleamos de ella y almacenamos su valor en el map
 */
public class BnSampler {

    public static DataOnMemory<DataInstance> sample(BayesianNetwork network,
                                                    int nSamples,
                                                    List<Variable> latentVariables,
                                                    Random random,
                                                    boolean returnLatentVarsData) {

        /* 1 - Sort variables usint their topological order */
        List<Variable> sortedVariables = GraphUtilsAmidst.topologicalSort(network.getDAG());

        /*
        * 2 - Create a new empty list of data instances with its corresponding list of attributes
        */
        List<DataInstance> dataInstanceList = new ArrayList<>();
        List<Attribute> attributeList = new ArrayList<>();
        Attributes attributes = new Attributes(attributeList);

        int sortedVarIndex = 0;
        for(Variable var: sortedVariables){
            if(!latentVariables.contains(var) || (latentVariables.contains(var) && returnLatentVarsData))
                attributeList.add(new Attribute(sortedVarIndex++, var.getName(), var.getStateSpaceType()));
        }

        Map<Variable, Double> currentVisitedVarsValues = new LinkedHashMap<>();
        /* 3 - Generate samples in order */
        for(int i = 0; i < nSamples; i++) {

            currentVisitedVarsValues.clear();
            double[] sampleValues = new double[attributeList.size()];
            int index = 0;

            for(Variable var: sortedVariables){

                ConditionalDistribution condDist = network.getConditionalDistribution(var);

                /* If the variable doesnt have any parents we directly sample from it */
                if(network.getDAG().getParentSet(var).getNumberOfParents() == 0) {
                    UnivariateDistribution univariateDist = (UnivariateDistribution) condDist;
                    double sampledValue = univariateDist.sample(random);
                    currentVisitedVarsValues.put(var, sampledValue);
                    if(!latentVariables.contains(var) || (latentVariables.contains(var) && returnLatentVarsData)) {
                        sampleValues[index] = sampledValue;
                        index++;
                    }

                } else {
                    List<Variable> parents = network.getDAG().getParentSet(var).getParents();
                    Assignment parentAssignment = new HashMapAssignment();
                    for(Variable parent: parents) {
                        double parentValue = currentVisitedVarsValues.get(parent);
                        parentAssignment.setValue(parent, parentValue);
                    }
                    UnivariateDistribution univariateDist = condDist.getUnivariateDistribution(parentAssignment);
                    double sampledValue = univariateDist.sample(random);
                    currentVisitedVarsValues.put(var, sampledValue);
                    if(!latentVariables.contains(var) || (latentVariables.contains(var) && returnLatentVarsData)) {
                        sampleValues[index] = sampledValue;
                        index++;
                    }
                }
            }
            dataInstanceList.add(new DataInstanceFromRawData(attributes, sampleValues));
        }

        return new DataOnMemoryListContainer<>(attributes, dataInstanceList);
    }

    public static DataOnMemory<DataInstance> newSample(BayesianNetwork network, int nSamples, Random random) {

        /* Generate a container for the sampled data instances and their attributes */
        List<DataInstance> dataInstanceList = new ArrayList<>();
        List<Attribute> attributeList = new ArrayList<>();
        Attributes attributes = new Attributes(attributeList);

        int attributeIndex = 0;
        for(Variable var: network.getVariables())
            attributeList.add(new Attribute(attributeIndex++, var.getName(), var.getStateSpaceType()));

        /* Store the index of the variables in a map */
        Map<Variable, Integer> mapVarIndex = new HashMap<>();
        for(Variable var: network.getVariables())
            mapVarIndex.put(var, network.getVariables().getListOfVariables().indexOf(var));

        /* Sort variables in topological order */
        List<Variable> sortedVariables = GraphUtilsAmidst.topologicalSort(network.getDAG());

        /* 3 - Generate samples in order */
        for(int i = 0; i < nSamples; i++){

            double[] sampledValues = new double[sortedVariables.size()];

            for (Variable var: sortedVariables) {

                ConditionalDistribution condDist = network.getConditionalDistribution(var);

                /* If the variable doesnt have any parents we directly sample from its conditional distribution */
                if(network.getDAG().getParentSet(var).getNumberOfParents() == 0) {
                    UnivariateDistribution univariateDist = (UnivariateDistribution) condDist;
                    double sampledValue = univariateDist.sample(random);
                    sampledValues[mapVarIndex.get(var)] = sampledValue;
                }

                /* Instantiate the parents and sample from the conditional distribution */
                else {
                    List<Variable> parents = network.getDAG().getParentSet(var).getParents();
                    Assignment parentAssignment = new HashMapAssignment();
                    for(Variable parent: parents) {
                        double parentValue = sampledValues[mapVarIndex.get(parent)];
                        parentAssignment.setValue(parent, parentValue);
                    }
                    UnivariateDistribution univariateDist = condDist.getUnivariateDistribution(parentAssignment);
                    double sampledValue = univariateDist.sample(random);
                    sampledValues[mapVarIndex.get(var)] = sampledValue;
                }
            }
            dataInstanceList.add(new DataInstanceFromRawData(attributes, sampledValues));
        }
        return new DataOnMemoryListContainer<>(attributes, dataInstanceList);
    }

}
