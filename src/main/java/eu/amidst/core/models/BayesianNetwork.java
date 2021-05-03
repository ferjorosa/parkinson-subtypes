/*
 *
 *
 *    Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.
 *    See the NOTICE file distributed with this work for additional information regarding copyright ownership.
 *    The ASF licenses this file to You under the Apache License, Version 2.0 (the "License"); you may not use
 *    this file except in compliance with the License.  You may obtain a copy of the License at
 *
 *            http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software distributed under the License is
 *    distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and limitations under the License.
 *
 *
 */

package eu.amidst.core.models;

import eu.amidst.core.distribution.*;
import eu.amidst.core.utils.Utils;
import eu.amidst.core.variables.Assignment;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.util.distance.DistanceFunction;
import eu.amidst.extension.util.distance.ManhattanDistance;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Random;

/**
 * The BayesianNetwork class represents a Bayesian network model.
 *
 * <p> For an example of use follow this link </p>
 * <p> <a href="http://amidst.github.io/toolbox/CodeExamples.html#bnexample"> http://amidst.github.io/toolbox/CodeExamples.html#bnexample </a>  </p>
 *
 * <p> For further details about the implementation of this class using Java 8 functional-style programming look at the following paper: </p>
 *
 * <i> Masegosa et al. Probabilistic Graphical Models on Multi-Core CPUs using Java 8. IEEE-CIM (2015). </i>
 *
 */
public final class BayesianNetwork implements Serializable {

    /** Represents the serial version ID for serializing the object. */
    private static final long serialVersionUID = 4107783324901370839L;

    /** Represents the list of conditional probability distributions defining the Bayesian network parameters. */
    private LinkedHashMap<Variable, ConditionalDistribution> distributions;

    /** Represents the Directed Acyclic Graph ({@link DAG}) defining the Bayesian network graphical structure. */
    private DAG dag;

    /**
     * Creates a new BayesianNetwork from a dag.
     * @param dag a directed acyclic graph.
     */
    public BayesianNetwork(DAG dag) {
        this.dag = dag;
        initializeDistributions();
    }

    /**
     * Creates a new BayesianNetwork from a dag and a list of distributions.
     * @param dag a directed acyclic graph.
     * @param dists a list of conditional probability distributions.
     */
    public BayesianNetwork(DAG dag, List<ConditionalDistribution> dists) {
        this.dag = dag;
        this.distributions = new LinkedHashMap<>();
        for(ConditionalDistribution dist: dists)
            this.distributions.put(dist.getVariable(), dist);
    }

    /**
     * Returns the name of the BN
     * @return a String object
     */
    public String getName() {
        return this.dag.getName();
    }

    /**
     * Sets the name of the BN
     * @param name, a String object
     */
    public void setName(String name) {
        this.dag.setName(name);
    }

    /**
     * Returns the conditional probability distribution of a variable.
     * @param <E> a class extending {@link ConditionalDistribution}.
     * @param var a variable of type {@link Variable}.
     * @return a conditional probability distribution.
     */
    public <E extends ConditionalDistribution> E getConditionalDistribution(Variable var) {
        return (E) distributions.get(var);
    }

    /**
     * Sets the conditional probability distribution of a variable.
     * @param var a variable of type {@link Variable}.
     * @param dist Conditional probability distribution of type {@link ConditionalDistribution}.
     */
    public void setConditionalDistribution(Variable var, ConditionalDistribution dist){
        this.distributions.put(var,dist);
    }

    /**
     * Returns the total number of variables in this BayesianNetwork.
     * @return the number of variables.
     */
    public int getNumberOfVars() {
        return this.getDAG().getVariables().getNumberOfVars();
    }

    /**
     * Returns the set of variables in this BayesianNetwork.
     * @return set of variables of type {@link Variables}.
     */
    public Variables getVariables() {
        return this.getDAG().getVariables();
    }

    /**
     * Returns the directed acyclic graph of this BayesianNetwork.
     * @return a directed acyclic graph of type {@link DAG}.
     */
    public DAG getDAG() {
        return dag;
    }

    /**
     * Returns the parameter values of this BayesianNetwork.
     * @return an array containing the parameter values of all distributions.
     */
    public double[] getParameters(){

        int size = this.distributions.values().stream().mapToInt(dist -> dist.getNumberOfParameters()).sum();

        double[] param = new double[size];

        int count = 0;

        for (Distribution dist : this.distributions.values()){
            System.arraycopy(dist.getParameters(), 0, param, count, dist.getNumberOfParameters());
            count+=dist.getNumberOfParameters();
        }

        return param;
    }

    public int getNumberOfParameters() {
        int count = 0;
        for (Distribution dist : this.distributions.values())
            count += dist.getNumberOfParameters();

        return count;
    }

    /**
     * Initializes the distributions of this BayesianNetwork.
     * The initialization is performed for each variable depending on its distribution type.
     * as well as the distribution type of its parent set (if that variable has parents).
     */
    private void initializeDistributions() {

        this.distributions = new LinkedHashMap<>();

        for (Variable var : getVariables()) {
            ParentSet parentSet = this.getDAG().getParentSet(var);
            this.distributions.put(var, var.newConditionalDistribution(parentSet.getParents()));
            parentSet.blockParents();
        }

        //this.distributions = Collections.unmodifiableList(this.distributions);
    }

    /**
     * Returns the log probability of a valid assignment.
     * @param assignment an object of type {@link Assignment}.
     * @return the log probability of an assignment.
     */
    public double getLogProbabiltyOf(Assignment assignment) {
        double logProb = 0;
        for (Variable var : this.getVariables()) {
            if (assignment.getValue(var) == Utils.missingValue()) {
                throw new UnsupportedOperationException("This method can not compute the probabilty of a partial assignment.");
            }

            logProb += this.distributions.get(var.getVarID()).getLogConditionalProbability(assignment);
        }
        return logProb;
    }

    /**
     * Returns the list of the conditional probability distributions of this BayesianNetwork.
     * @return a list of {@link ConditionalDistribution}.
     */
    public List<ConditionalDistribution> getConditionalDistributions() {
        return new ArrayList<>(this.distributions.values());
    }

    /**
     * Returns a textual representation of this BayesianNetwork.
     * @return a String description of this BayesianNetwork.
     */
    public String toString() {

        StringBuilder str = new StringBuilder();
        str.append("Bayesian Network:\n");

        for (Variable var : this.getVariables()) {

            if (this.getDAG().getParentSet(var).getNumberOfParents() == 0) {
                str.append("P(" + var.getName() + ") follows a ");
                str.append(this.getConditionalDistribution(var).label() + "\n");
            } else {
                str.append("P(" + var.getName() + " | ");

                for (Variable parent : this.getDAG().getParentSet(var)) {
                    str.append(parent.getName() + ", ");
                }
                str.delete(str.length()-2,str.length());
                if (this.getDAG().getParentSet(var).getNumberOfParents() > 0) {
                    str.substring(0, str.length() - 2);
                    str.append(") follows a ");
                    str.append(this.getConditionalDistribution(var).label() + "\n");
                }
            }
            //Variable distribution
            str.append(this.getConditionalDistribution(var).toString() + "\n");
        }
        return str.toString();
    }

    /**
     * Initializes the distributions of this BayesianNetwork randomly.
     * @param random an object of type {@link Random}.
     */
    public void randomInitialization(Random random) {
        this.distributions.values().stream().forEach(w -> w.randomInitialization(random));
    }

    /**
     * Randomly initializes conditional distributions. It uses an internal heuristic to avoid independences (manhattan
     * distance with a threshold). It will try to generate a parameter vector with the specific distance threshold (to all
     * the other parametrizations), if it hasnt been possible after nTries, it will set the closest one.
     *
     * @param random
     * @param multinomialThreshold
     * @param normalThreshold
     * @param nTries
     */
    public void randomTreeInitialization(Random random, double multinomialThreshold, double normalThreshold, int nTries) {

        DistanceFunction distanceFunction = new ManhattanDistance();

        for(ConditionalDistribution condDist: this.distributions.values()){

            // Univariate distributions & CLGs
            if(condDist instanceof Multinomial || condDist instanceof Normal || condDist instanceof ConditionalLinearGaussian) {
                condDist.randomInitialization(random);

            // Multinomial - Multinomial
            } else if(condDist instanceof Multinomial_MultinomialParents) {

                Multinomial_MultinomialParents condMultinomial = (Multinomial_MultinomialParents) condDist;
                int parentStates = condDist.getParents().get(0).getNumberOfStates();

                /* Iterate through all the parent combinations */
                for(int i = 0; i < parentStates; i++) {

                    Multinomial multinomial = condMultinomial.getMultinomial(i);
                    double[] bestParameters = new double[condDist.getVariable().getNumberOfStates()];
                    double bestDistance = 0;
                    int currentTries = 0;

                    /*
                     * Keep generating random parameters until their value difference with the other normal parameters
                     * is greater than multinomialThreshold or until the number of tries has been exceeded. In that case,
                     * we simply assign the most different parametrization.
                     */
                    boolean similarCondParameter = false;
                    do {
                        currentTries++;
                        multinomial.randomInitialization(random);
                        double[] currentParameters = multinomial.getParameters();
                        /* Measure the Manhattan distance with all previous conditional-multinomial parameters */
                        for(int j = 0; j < i; j++){

                            double currentDistance = distanceFunction.distance(currentParameters, condMultinomial.getMultinomial(j).getParameters());

                            if(currentDistance < multinomialThreshold) {
                                similarCondParameter = true;
                                if(currentDistance > bestDistance) {
                                    bestParameters = currentParameters;
                                    bestDistance = currentDistance;
                                }
                            } else
                                similarCondParameter = false;
                        }

                    } while(similarCondParameter && (currentTries < nTries));

                    if(currentTries >= nTries)
                        multinomial.setProbabilities(bestParameters);
                }

            // Normal - Multinomial
            } else if(condDist instanceof Normal_MultinomialParents) {
                Normal_MultinomialParents condNormal = (Normal_MultinomialParents) condDist;
                int parentStates = condDist.getParents().get(0).getNumberOfStates();

                /* Iterate through all the parent combinations */
                for(int i = 0; i < parentStates; i++) {

                    Normal normal = condNormal.getNormal(i);
                    double[] bestParameters = new double[2];
                    double bestDistance = 0;
                    int currentTries = 0;

                    /*
                     * Keep generating random parameters until their value difference with the other normal parameters
                     * is greater than normalThreshold or until the number of tries has been exceeded. In that case,
                     * we simply assign the most different parametrization.
                     */
                    boolean similarCondParameter = false;
                    do {
                        currentTries++;
                        normal.randomInitialization(random);
                        double[] currentParameters = normal.getParameters();
                        /* Measure the Manhattan distance with all previous conditional-multinomial parameters */
                        for(int j = 0; j < i; j++){

                            double currentDistance = distanceFunction.distance(currentParameters, condNormal.getNormal(j).getParameters());

                            if(currentDistance < normalThreshold) {
                                similarCondParameter = true;
                                if(currentDistance > bestDistance) {
                                    bestParameters = currentParameters;
                                    bestDistance = currentDistance;
                                }
                            } else
                                similarCondParameter = false;
                        }

                    } while(similarCondParameter && (currentTries < nTries));

                    if(currentTries >= nTries) {
                        normal.setMean(bestParameters[0]);
                        normal.setVariance(bestParameters[1]);
                    }
                }
            }
        }
    }

    /**
     * Tests if two Bayesian networks are equals.
     * A two Bayesian networks are considered equals if they have an equal conditional distribution for each variable.
     * @param bnet a given BayesianNetwork to be compared with this BayesianNetwork.
     * @param threshold a threshold value.
     * @return a boolean indicating if the two BNs are equals or not.
     */
    public boolean equalBNs(BayesianNetwork bnet, double threshold) {
        boolean equals = true;
        if (this.getDAG().equals(bnet.getDAG())){
            for (Variable var : this.getVariables()) {
                equals = equals && this.getConditionalDistribution(var).equalDist(bnet.getConditionalDistribution(var), threshold);
            }
        }
        return equals;
    }

    /**
     * Returns this class name.
     * @return a String representing this class name.
     */
    public static String listOptions() {
        return  classNameID();
    }

    public static String listOptionsRecursively() {
        return listOptions()
                + "\n" +  "test";
    }

    public static String classNameID() {
        return "BayesianNetwork";
    }

    public static void loadOptions() {

    }
}

