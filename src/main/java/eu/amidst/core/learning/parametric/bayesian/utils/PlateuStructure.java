/*
 * Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership. The ASF licenses this file to You under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 * See the License for the specific language governing permissions and limitations under the License.
 *
 */

package eu.amidst.core.learning.parametric.bayesian.utils;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.exponentialfamily.EF_ConditionalDistribution;
import eu.amidst.core.exponentialfamily.EF_LearningBayesianNetwork;
import eu.amidst.core.exponentialfamily.EF_UnivariateDistribution;
import eu.amidst.core.exponentialfamily.NaturalParameters;
import eu.amidst.core.inference.messagepassing.Node;
import eu.amidst.core.inference.messagepassing.VMP;
import eu.amidst.core.models.DAG;
import eu.amidst.core.utils.CompoundVector;
import eu.amidst.core.utils.Vector;
import eu.amidst.core.variables.Variable;
import eu.amidst.extension.util.tuple.Tuple2;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * This class defines a Plateu Structure.
 */
public abstract class PlateuStructure implements Serializable {

    /**
     * Represents the serial version ID for serializing the object.
     */
    private static final long serialVersionUID = 4107783324901370839L;

    /**
     * Represents a map describing which variables are replicated
     */
    protected Map<Variable, Boolean> replicatedVariables = new HashMap<>();

    /**
     * Represents the list of non replicated {@link Node}s.
     */
    transient protected List<Node> nonReplicatedNodes = new ArrayList();

    /**
     * Represents the list of replicated nodes {@link Node}s.
     */
    transient protected List<List<Node>> replicatedNodes = new ArrayList<>();

    /**
     * Represents the {@link EF_LearningBayesianNetwork} model.
     */
    protected EF_LearningBayesianNetwork ef_learningmodel;

    /**
     * Represents the number of replications.
     */
    protected int nReplications = 100;

    /**
     * Represents the {@link VMP} object.
     */
    protected VMP vmp = new VMP();

    /**
     * Represents a {@code Map} object that maps {@link Variable} parameters to the corresponding {@link Node}s.
     */
    transient protected Map<Variable, Node> nonReplicatedVarsToNode = new ConcurrentHashMap<>();

    /**
     * Represents the list of {@code Map} objects that map {@link Variable}s to the corresponding {@link Node}s.
     */
    transient protected List<Map<Variable, Node>> replicatedVarsToNode = new ArrayList<>();


    /**
     * Represents the initial list of non-replicated variables
     */
    protected List<Variable> initialNonReplicatedVariablesList;


    /**
     * Represents the list of non-replicated variables
     */
    protected List<Variable> nonReplicatedVariablesList;


    /**
     * Empty builder.
     */
    public PlateuStructure() {
        initialNonReplicatedVariablesList = new ArrayList<>();
    }

    /**
     * Builder which initially specify a list of non-replicated variables.
     *
     * @param initialNonReplicatedVariablesList list of variables
     */
    public PlateuStructure(List<Variable> initialNonReplicatedVariablesList) {
        this.initialNonReplicatedVariablesList = new ArrayList<>();
        this.initialNonReplicatedVariablesList.addAll(initialNonReplicatedVariablesList);
    }

    /**
     * Initializes the interal transient data structures.
     */
    public void initTransientDataStructure(){
        replicatedVarsToNode = new ArrayList<>();
        nonReplicatedVarsToNode = new ConcurrentHashMap<>();
        replicatedNodes = new ArrayList<>();
        nonReplicatedNodes = new ArrayList();
    }

    public Stream<Node> getNonReplicatedNodes() {
        return nonReplicatedNodes.stream();
    }

    public Stream<Node> getReplicatedNodes() {
        return replicatedNodes.stream().flatMap(l -> l.stream());
    }

    /**
     * Returns the number of replications of this PlateuStructure.
     *
     * @return the number of replications.
     */
    public int getNumberOfReplications() {
        return nReplications;
    }

    /**
     * Returns the {@link VMP} object of this PlateuStructure.
     *
     * @return the {@link VMP} object.
     */
    public VMP getVMP() {
        return vmp;
    }


    public void setVmp(VMP vmp) {
        this.vmp = vmp;
    }

    /**
     * Resets the exponential family distributions of all nodes for the {@link VMP} object of this PlateuStructure.
     */
    public void resetQs() {
        this.vmp.resetQs();
    }

    public void resetQs(List<Node> nodes) {
        this.vmp.resetQs(nodes);
    }

    /**
     * Sets the seed for the {@link VMP} object of this PlateuStructure.
     *
     * @param seed an {@code int} that represents the seed value.
     */
    public void setSeed(long seed) {
        this.vmp.setSeed(seed);
    }

    /**
     * Returns the {@link EF_LearningBayesianNetwork} of this PlateuStructure.
     *
     * @return an {@link EF_LearningBayesianNetwork} object.
     */
    public EF_LearningBayesianNetwork getEFLearningBN() {
        return ef_learningmodel;
    }

    /**
     * Sets the {@link DAG} of this PlateuStructure. By default,
     * all parameter variables are set as non-replicated and all non-parameter variables
     * are set as replicated.
     *
     * @param dag the {@link DAG} model to be set.
     */
    public void setDAG(DAG dag) {

        List<EF_ConditionalDistribution> dists = dag.getParentSets().stream()
                .map(pSet -> pSet.getMainVar().getDistributionType().<EF_ConditionalDistribution>newEFConditionalDistribution(pSet.getParents()))
                .collect(Collectors.toList());

        ef_learningmodel = new EF_LearningBayesianNetwork(dists, this.initialNonReplicatedVariablesList);
        this.ef_learningmodel.getListOfParametersVariables().stream().forEach(var -> this.replicatedVariables.put(var, false));
        this.ef_learningmodel.getListOfNonParameterVariables().stream().forEach(var -> this.replicatedVariables.put(var, true));

        this.initialNonReplicatedVariablesList.stream().forEach(var -> this.replicatedVariables.put(var, false));


        this.nonReplicatedVariablesList = this.replicatedVariables.entrySet().stream().filter(entry -> !entry.getValue()).map(entry -> entry.getKey()).sorted((a,b) -> a.getVarID()-b.getVarID()).collect(Collectors.toList());
    }

    /**
     * Sets the {@link DAG} of this PlateuStructure. By default, all parameter variables are set as non-replicated
     * and all non-parameter variables are set as replicated.
     *
     * In addition, if there is no associated prior, it sets the default prior to the parameter variables.
     *
     * @param dag the {@link DAG} model to be set.
     */
    /*
        MyNote: Por el momento no distingo dentro de una mixtura, le asigno la misma prior a una variable independientemente
        del cluster al que pertenezca.
     */
    public void setDAG(DAG dag, Map<String, double[]> priorsParameters) {

        List<EF_ConditionalDistribution> dists = dag.getParentSets().stream()
                .map(pSet -> pSet.getMainVar().getDistributionType().<EF_ConditionalDistribution>newEFConditionalDistribution(pSet.getParents()))
                .collect(Collectors.toList());

        ef_learningmodel = new EF_LearningBayesianNetwork(dists, this.initialNonReplicatedVariablesList, priorsParameters);
        this.ef_learningmodel.getListOfParametersVariables().stream().forEach(var -> this.replicatedVariables.put(var, false));
        this.ef_learningmodel.getListOfNonParameterVariables().stream().forEach(var -> this.replicatedVariables.put(var, true));

        this.initialNonReplicatedVariablesList.stream().forEach(var -> this.replicatedVariables.put(var, false));


        this.nonReplicatedVariablesList = this.replicatedVariables.entrySet().stream().filter(entry -> !entry.getValue()).map(entry -> entry.getKey()).sorted((a,b) -> a.getVarID()-b.getVarID()).collect(Collectors.toList());
    }

    /**
     * Sets a given variable as a non replicated variable.
     *
     * @param var, a {@link Variable} object.
     */
    private void setVariableAsNonReplicated(Variable var) {
        this.replicatedVariables.put(var, false);
    }

    /**
     * Sets a given variable as a replicated variable.
     *
     * @param var, a {@link Variable} object.
     */
    private void setVariableAsReplicated(Variable var) {
        this.replicatedVariables.put(var, true);
    }

    /**
     * Returns the list of non replicated Variables
     *
     * @return list of variables
     */
    public List<Variable> getNonReplicatedVariables() {
        return this.nonReplicatedVariablesList;
    }


    public List<Variable> getReplicatedVariables() {
        return this.replicatedVariables.entrySet().stream().filter(entry -> entry.getValue()).map(entry -> entry.getKey()).sorted((a,b) -> a.getVarID()-b.getVarID()).collect(Collectors.toList());
    }

    /**
     * Sets the number of repetitions for this PlateuStructure.
     *
     * @param nRepetitions_ an {@code int} that represents the number of repetitions to be set.
     */
    public void setNRepetitions(int nRepetitions_) {
        this.nReplications = nRepetitions_;
    }

    /** Runs inference. */
    public void runInference() { this.vmp.runInference(); }

    /* ****************************************************************************************************************/

    public void emInference() { this.vmp.emInference(); }

    public void emInference(List<Node> nodes) { this.vmp.emInference(nodes); }

    public void emInference(int nIterations) { this.vmp.emInference(nIterations); }

    public void emInference(List<Node> nodes, int nIterations) { this.vmp.emInference(nodes, nIterations); }

    public void emInferenceWithoutConvergence(int nIterations) { this.vmp.emInferenceWithoutConvergence(nIterations); }

    public void emInferenceWithoutConvergence(List<Node> nodes, int nIterations) { this.vmp.emInferenceWithoutConvergence(nodes, nIterations); }

    /* ****************************************************************************************************************/

    // TODO: En principio vamos a probar a no pasarle el conjunto de nodos de la variable en cuestion pero si calcular solo su score
    public void runInferenceHC(List<Node> nodes) {
        this.vmp.runInferenceHC(nodes);
    }

    /**
     * Devuelve el score de cada variable "repetida" agrupando el score suyo propio y de sus parameter variables
     */
    public Map<Variable, Double> computeLogProbabilityOfEvidenceMap() {

        // TODO: Quizas podemos calcular el score Map directamente aqui en vez de hacerlo en VMP para no añadir mas codigo
        /* Obtenemos un Map con los scores de todas las variables, incluidas parameter variables */
        Map<Variable, Double> map = this.vmp.computeLogProbabilityOfEvidenceMap();
        Map<Variable, Double> newMap = new HashMap<>();
        for(Variable replicatedVar: replicatedVariables.keySet())
            if(replicatedVariables.get(replicatedVar))//Check it is replicated
                newMap.put(replicatedVar, 0.0);

        // Miramos dentro de EF_Learning_BN el conjunto de distribuciones, iteramos por cada una de ellas e iteramos sus padres
        // Si el padre es una variable parametro

        for(EF_ConditionalDistribution dist: this.ef_learningmodel.getDistributionList().values()){
            double score = map.get(dist.getVariable());
            // Si tiene padres que sean parameter-variables, añadimos su score correspondiente
            for(Variable parentVar: dist.getConditioningVariables()){
                if(parentVar.isParameterVariable()){
                    double parameterVarScore = map.get(parentVar);
                    score += parameterVarScore;
                    newMap.put(dist.getVariable(), score);
                }
            }
        }

        return newMap;
    }

    /**
     * Utiliza una Tuple con lista, la cual requiere que se mantenga el orden o no valdrá.
     * Por ello, vamos a crear la version 3, que utilice un HashSet, sino tendriamos que estar controlando
     * el orden en muchos sitios para que el equals de dicha tuple fuera correcto.
     */
    public Map<Tuple2<Variable, List<Variable>>, Double> computeLogProbabilityOfEvidenceMap_2() {

        /* OGenerate a Map with variables' scores, including parameter variables */
        // TODO: Quizas podemos calcular el score Map directamente aqui en vez de hacerlo en VMP para no añadir mas codigo
        Map<Variable, Double> map = this.vmp.computeLogProbabilityOfEvidenceMap();

        /* Create a new Map to store the scores of non-parameter variables combined with the scores of its parameter variables */
        Map<Tuple2<Variable, List<Variable>>, Double> newMap = new HashMap<>();

        /* Iterate through distributions to find the parameter variables */
        for(EF_ConditionalDistribution dist: this.ef_learningmodel.getDistributionList().values()){

            Variable var = dist.getVariable();

            if(!var.isParameterVariable()) {

                /* Start with the variable's score */
                double score = map.get(var);
                List<Variable> nonParameterParents = new ArrayList<>();

                /* Iterate through its parent list and select parameter variables */
                for (Variable parentVar : dist.getConditioningVariables()) {
                    if (parentVar.isParameterVariable()) {
                        double parameterVarScore = map.get(parentVar);
                        score += parameterVarScore;
                    }
                    /* If its a nonParameterVariable, add it to the list */
                    else
                        nonParameterParents.add(parentVar);
                }
                /* Create a new Tuple and add it to the Map */
                newMap.put(new Tuple2<>(var, nonParameterParents), score);
            }
        }

        return newMap;
    }

    /**
     *
     */
    public Map<Tuple2<Variable, Set<Variable>>, Double> computeLogProbabilityOfEvidenceMap_3() {

        /* OGenerate a Map with variables' scores, including parameter variables */
        // TODO: Quizas podemos calcular el score Map directamente aqui en vez de hacerlo en VMP para no añadir mas codigo
        Map<Variable, Double> map = this.vmp.computeLogProbabilityOfEvidenceMap();

        /* Create a new Map to store the scores of non-parameter variables combined with the scores of its parameter variables */
        Map<Tuple2<Variable, Set<Variable>>, Double> newMap = new HashMap<>();

        /* Iterate through distributions to find the parameter variables */
        for(EF_ConditionalDistribution dist: this.ef_learningmodel.getDistributionList().values()){

            Variable var = dist.getVariable();

            if(!var.isParameterVariable()) {

                /* Start with the variable's score */
                double score = map.get(var);
                Set<Variable> nonParameterParents = new HashSet<>();

                /* Iterate through its parent list and select parameter variables */
                for (Variable parentVar : dist.getConditioningVariables()) {
                    if (parentVar.isParameterVariable()) {
                        double parameterVarScore = map.get(parentVar);
                        score += parameterVarScore;
                    }
                    /* If its a nonParameterVariable, add it to the list */
                    else
                        nonParameterParents.add(parentVar);
                }
                /* Create a new Tuple and add it to the Map */
                newMap.put(new Tuple2<>(var, nonParameterParents), score);
            }
        }

        return newMap;
    }

    /**
     * Returns the log probability of the evidence.
     *
     * @return the log probability of the evidence.
     */
    public double getLogProbabilityOfEvidence() {
        return this.vmp.getLogProbabilityOfEvidence();
    }

    /**
     * Returns the {@link Node} for a given variable and slice.
     *
     * @param variable a {@link Variable} object.
     * @param slice    an {@code int} that represents the slice value.
     * @return a {@link Node} object.
     */
    public Node getNodeOfVar(Variable variable, int slice) {
        if (isNonReplicatedVar(variable))
            return this.nonReplicatedVarsToNode.get(variable);
        else
            return this.replicatedVarsToNode.get(slice).get(variable);
    }

    /**
     * Returns the exponential family parameter posterior for a given {@link Variable} object.
     *
     * @param var a given {@link Variable} object.
     * @param <E> a subtype distribution of {@link EF_UnivariateDistribution}.
     * @return an {@link EF_UnivariateDistribution} object.
     */
    public <E extends EF_UnivariateDistribution> E getEFParameterPosterior(Variable var) {
        if (!this.nonReplicatedVariablesList.contains(var) && !var.isParameterVariable())
            throw new IllegalArgumentException("Only non replicated variables or parameters can be queried");

        return (E) this.nonReplicatedVarsToNode.get(var).getQDist();
    }

    /**
     * Returns the exponential family variable posterior for a given {@link Variable} object and a slice value.
     *
     * @param var   a given {@link Variable} object.
     * @param slice an {@code int} that represents the slice value.
     * @param <E>   a subtype distribution of {@link EF_UnivariateDistribution}.
     * @return an {@link EF_UnivariateDistribution} object.
     */
    public <E extends EF_UnivariateDistribution> E getEFVariablePosterior(Variable var, int slice) {
        if (this.nonReplicatedVariablesList.contains(var) || var.isParameterVariable())
            throw new IllegalArgumentException("Only replicated variables can be queried");

        return (E) this.getNodeOfVar(var, slice).getQDist();
    }

    /**
     * Replicates the model of this PlateuStructure.
     */
    public abstract void replicateModel();

    /**
     * Sets the evidence for this PlateuStructure.
     *
     * @param data a {@code List} of {@link DataInstance}.
     */
    public abstract void setEvidence(List<? extends DataInstance> data);

    public Node getNodeOfNonReplicatedVar(Variable variable) {
        if (isNonReplicatedVar(variable))
            return this.nonReplicatedVarsToNode.get(variable);
        else
            throw new IllegalArgumentException("This variable is a replicated var.");
    }

    public boolean isNonReplicatedVar(Variable var){
        return !this.replicatedVariables.get(var);
    }

    public boolean isReplicatedVar(Variable var){
        return this.replicatedVariables.get(var);
    }

    /* MyNote: En vez de tener 3 metodos de get y de update se podria reducir a 1 de cada si se le permite pasar un argumento */
    public CompoundVector getParameterVariablePriors() {

        List<Vector> naturalPlateauParametersPriors = ef_learningmodel.getDistributionList().values().stream()
                .map(dist -> dist.getVariable())
                .filter(var -> isNonReplicatedVar(var))
                .map(var -> {
                    NaturalParameters parameter = this.ef_learningmodel.getDistribution(var).getNaturalParameters();
                    NaturalParameters copy = this.ef_learningmodel.getDistribution(var).createZeroNaturalParameters();
                    copy.copy(parameter);
                    return copy;
                }).collect(Collectors.toList());

        return new CompoundVector(naturalPlateauParametersPriors);
    }

    /* MyNote: Este es el metodo que devuelve las posteriors aprendidas de las variables no replicadas */
    public CompoundVector getParameterVariablesPosterior() {

        List<Vector> naturalPlateauParametersPosteriors = ef_learningmodel.getDistributionList().values().stream()
                .map(dist -> dist.getVariable())
                .filter(var -> isNonReplicatedVar(var))
                .map(var -> {
                    EF_UnivariateDistribution qDist = this.getNodeOfNonReplicatedVar(var).getQDist();
                    NaturalParameters parameter = qDist.getNaturalParameters();
                    NaturalParameters copy = qDist.createZeroNaturalParameters();
                    copy.copy(parameter);
                    return copy;
                }).collect(Collectors.toList());

        return new CompoundVector(naturalPlateauParametersPosteriors);
    }

    public void updateParameterVariablesPosterior(CompoundVector parameterVector) {

        final int[] count = new int[1];
        count[0] = 0;

        ef_learningmodel.getDistributionList().values().stream()
                .map(dist -> dist.getVariable())
                .filter(var -> isNonReplicatedVar(var))
                .forEach(var -> {
                    EF_UnivariateDistribution uni = this.getNodeOfNonReplicatedVar(var).getQDist();
                    uni.getNaturalParameters().copy(parameterVector.getVectorByPosition(count[0]));
                    uni.fixNumericalInstability();
                    uni.updateMomentFromNaturalParameters();
                    count[0]++;
                });
    }

    /**
     * Updates the Natural Parameter Prior from a given parameter vector.
     * @param parameterVector a {@link CompoundVector} object.
     */
    /* MyNote: Se hace al terminar de aprender con VBEM/SVB */
    public void updateParameterVariablesPrior(CompoundVector parameterVector) {

        final int[] count = new int[1];
        count[0] = 0;

        ef_learningmodel.getDistributionList().values().stream()
                .map(dist -> dist.getVariable())
                .filter(var -> isNonReplicatedVar(var))
                .forEach(var -> {
                    EF_UnivariateDistribution uni = this.getNodeOfNonReplicatedVar(var).getQDist().deepCopy();
                    uni.getNaturalParameters().copy(parameterVector.getVectorByPosition(count[0]));
                    uni.fixNumericalInstability();
                    uni.updateMomentFromNaturalParameters();
                    this.ef_learningmodel.setDistribution(var, uni);
                    this.getNodeOfNonReplicatedVar(var).setPDist(uni);
                    count[0]++;
                });
    }

    /**
     * Returns the posteriors of all the latent variables in the model (not just the parameter variables).
     * @return
     */
    public CompoundVector getLatentVariablesPosterior() {

        List<Vector> posteriors = this.vmp.getNodes().stream()
                .filter(node-> !node.isObserved())
                .map(node -> {
                    EF_UnivariateDistribution qDist = node.getQDist();
                    NaturalParameters parameter = qDist.getNaturalParameters();
                    NaturalParameters copy = qDist.createZeroNaturalParameters();
                    copy.copy(parameter);
                    return copy;
                }).collect(Collectors.toList());

        return new CompoundVector(posteriors);
    }


    public CompoundVector getLatentVariablesPosterior(List<Node> nodes) {
        List<Vector> posteriors = nodes.stream()
                .filter(node-> !node.isObserved())
                .map(node -> {
                    EF_UnivariateDistribution qDist = node.getQDist();
                    NaturalParameters parameter = qDist.getNaturalParameters();
                    NaturalParameters copy = qDist.createZeroNaturalParameters();
                    copy.copy(parameter);
                    return copy;
                }).collect(Collectors.toList());

        return new CompoundVector(posteriors);
    }

    /**
     * Updates the posteriors of all the latent variables in the model (not just the parameter variables).
     * @param parameterVector
     */
    /* MyNote: Lo utilizo dentro de Chickering & Heckerman */
    public void updateLatentVariablesPosterior(CompoundVector parameterVector) {

        final int[] count = new int[1];
        count[0] = 0;

        this.vmp.getNodes().stream()
                .filter(node-> !node.isObserved())
                .forEach(node -> {
                    EF_UnivariateDistribution uni = node.getQDist();
                    uni.getNaturalParameters().copy(parameterVector.getVectorByPosition(count[0]));
                    uni.fixNumericalInstability();
                    uni.updateMomentFromNaturalParameters();
                    count[0]++;
                });
    }

    public void updateLatentVariablesPosterior(List<Node> nodes, CompoundVector parameterVector) {
        final int[] count = new int[1];
        count[0] = 0;

        nodes.stream()
                .filter(node-> !node.isObserved())
                .forEach(node -> {
                    EF_UnivariateDistribution uni = node.getQDist();
                    uni.getNaturalParameters().copy(parameterVector.getVectorByPosition(count[0]));
                    uni.fixNumericalInstability();
                    uni.updateMomentFromNaturalParameters();
                    count[0]++;
                });
    }

    public void desactiveParametersNodes(){
        this.ef_learningmodel.getParametersVariables().getListOfParamaterVariables().stream()
                .forEach(var -> this.getNodeOfNonReplicatedVar(var).setActive(false));
    }

    public void activeParametersNodes() {
        this.ef_learningmodel.getParametersVariables().getListOfParamaterVariables().stream()
                .forEach(var -> this.getNodeOfNonReplicatedVar(var).setActive(true));
    }

    public double getPosteriorSampleSize(){
        return Double.NaN;
    }

    /**
     * Devuelve los nodos latentes asocidados a la variable argumento. Esto incluye sus propios nodos replicados en caso
     * de que ella sea latente e independientemente, sus parameter nodes padres.
     *
     * MYNote: Pasarle una variable replicada, no una parameterVariable
     *
     * @param variable
     * @return
     */
    public List<Node> getLatentNodes(Variable variable) {

        List<Node> latentNodes = new ArrayList<>();

        /* Si la variable es latente, añadimos sus nodos replicados */
        if(variable.getAttribute() == null){
            for(Map<Variable, Node> mapVarToNode: this.replicatedVarsToNode)
                latentNodes.add(mapVarToNode.get(variable));
        }

        /* Independientemente de si es latente o no, añadimos sus parameter variable nodes (los cuales son latentes) */
        List<Node> parameterVariableNodes = this.ef_learningmodel.getDistribution(variable)
                .getConditioningVariables()
                .stream()
                .filter(Variable::isParameterVariable)
                .map(x->this.nonReplicatedVarsToNode.get(x))
                .collect(Collectors.toList());

        latentNodes.addAll(parameterVariableNodes);

        return latentNodes;
    }

    public List<Node> getNodes(Variable variable) {

        List<Node> nodes = new ArrayList<>();
        for(Map<Variable, Node> mapVarToNode: this.replicatedVarsToNode)
            nodes.add(mapVarToNode.get(variable));

        List<Node> parameterVariableNodes = this.ef_learningmodel.getDistribution(variable)
                .getConditioningVariables()
                .stream()
                .filter(Variable::isParameterVariable)
                .map(x->this.nonReplicatedVarsToNode.get(x))
                .collect(Collectors.toList());

        nodes.addAll(parameterVariableNodes);

        return nodes;
    }

    /** Copy method. Classic one. Not much use yet */
    public abstract PlateuStructure deepCopy(DAG dag);

    /** Copy method. It is used in structure learning to generate a new Plateau with certain parts being copied */
    public abstract PlateuStructure deepCopy(DAG dag, Set<Variable> omittedVariables);
}