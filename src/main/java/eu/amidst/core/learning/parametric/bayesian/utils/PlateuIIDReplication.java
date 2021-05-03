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
import eu.amidst.core.inference.messagepassing.Node;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * This class extends the abstract class {@link PlateuStructure} and defines Plateu IID Replication.
 */
public class PlateuIIDReplication extends PlateuStructure{

    /**
     * Empty builder.
     */
    public PlateuIIDReplication() {
        super();
    }

    /**
     * Builder which initially specify a list of non-replicated variables.
     *
     * @param initialNonReplicatedVariablesList list of variables
     */
    public PlateuIIDReplication(List<Variable> initialNonReplicatedVariablesList) {
        super(initialNonReplicatedVariablesList);
    }

    /**
     * Crea una copia del Plateau con los valores del viejo excepto para la variable seleccionada y sus hijos.
     *
     * Se usa actualmente al incrementar la cardinalidad de una latente discreta
     */
    public PlateuIIDReplication(PlateuIIDReplication plateuIIDReplication, DAG dag, Set<Variable> omittedVariables) {
        /* initTransientStructure */
        this.replicatedVarsToNode = new ArrayList<>();
        this.nonReplicatedVarsToNode = new ConcurrentHashMap<>();
        this.replicatedNodes = new ArrayList<>();
        this.nonReplicatedNodes = new ArrayList<>();
        this.nReplications = plateuIIDReplication.nReplications;

        /* Le asignamos el DAG copia */
        this.setDAG(dag);

        /* Replicamos el modelo */
        this.replicateModel();

        /*
         * Filtramos todos los nodos no-replicados cuya non-parameter variable se encuentre en omittedVariables.
         *
         * MyNote: No podemos utilizar el HashMap "replicatedVarsToNode" ya que si hay diferente cardinalidad los varIDs
         * y nombres de las parameter variables cambian y con ello sus hashCodes.
         *
         * MyNote: Solo consideramos el primer hijo porque un parameter node solo puede ser padre de non-parameter nodes
         * donde todos ellos poseen la misma non-parameter variable hija
         */
        /*
        List<Node> filteredNonReplicatedNodes = plateuIIDReplication.nonReplicatedNodes.stream()
                .filter(node -> !omittedVariables.contains(node.getChildren().get(0).getMainVariable()))
                .collect(Collectors.toList());

        List<Node> copyFilteredNonReplicatedNodes =this.nonReplicatedNodes.stream()
                .filter(node -> !omittedVariables.contains(node.getChildren().get(0).getMainVariable()))
                .collect(Collectors.toList());
        */
        /*
            Copiamos el contenido de los nodos no replicados ya filtrados. Iteramos con indice porque deberian tener el
            mismo tamaño y orden
         */
        /*
        for(int i = 0; i < filteredNonReplicatedNodes.size(); i++) {
            Node node = filteredNonReplicatedNodes.get(i);
            Node copy = copyFilteredNonReplicatedNodes.get(i);

            for(int j = 0; j < node.getQDist().getNaturalParameters().size(); j++) {
                copy.getQDist().getNaturalParameters().set(j, node.getQDist().getNaturalParameters().get(j));
                copy.getQDist().getMomentParameters().set(j, node.getQDist().getMomentParameters().get(j));

                copy.getPDist().getNaturalParameters().set(j, node.getPDist().getNaturalParameters().get(j));
                copy.getPDist().getMomentParameters().set(j, node.getPDist().getMomentParameters().get(j));
            }
        }
        */

        /*
        * Filtramos todos los nodos no replicados cuya non-parameter variable se encuentra en omittedVariables.
        *
        * MyNote: No podemos utilizar el HashMap "replicatedVarsToNode" ya que si hay diferente cardinalidad los varIDs
        * y nombres de las parameter variables cambian y con ello sus hashCodes. Por ello, utilizamos el nombre sin indice
        * que nos permite relacionar nodos con diferente ID y nombre completo.
        *
        * MyNote: Solo consideramos el primer hijo porque un parameter node solo puede ser padre de non-parameter nodes
        * donde todos ellos poseen la misma non-parameter variable hija
        */
        Map<String, Node> nameToNode = plateuIIDReplication.nonReplicatedNodes.stream()
                .filter(node -> !omittedVariables.contains(node.getChildren().get(0).getMainVariable()))
                .collect(Collectors.toMap(node -> removeIndexFromName(node.getName()), node -> node));

        Map<String, Node> nameToCopy = this.nonReplicatedNodes.stream()
                .filter(node -> !omittedVariables.contains(node.getChildren().get(0).getMainVariable()))
                .collect(Collectors.toMap(node -> removeIndexFromName(node.getName()), node -> node));

        /*
         * Copiamos todos los nodos no replicados que no se encuentren omitidos. Para evitar problemas comprobamos que
         * el nodo a copiar se encuentre en el de inicio, sino realmente deberia estar omitido...
         */
        for(String nodeName: nameToNode.keySet()){

            Node node = nameToNode.get(nodeName);
            Node copy = nameToCopy.get(nodeName);

            for(int i = 0; i < node.getQDist().getNaturalParameters().size(); i++) {
                copy.getQDist().getNaturalParameters().set(i, node.getQDist().getNaturalParameters().get(i));
                copy.getQDist().getMomentParameters().set(i, node.getQDist().getMomentParameters().get(i));

                copy.getPDist().getNaturalParameters().set(i, node.getPDist().getNaturalParameters().get(i));
                copy.getPDist().getMomentParameters().set(i, node.getPDist().getMomentParameters().get(i));
            }
        }

        /*
        * Copiamos todos los nodos replicados excepto los latentes que esten en omittedVariables. Para evitar problemas,
        * comprobamos que los nodos esten en ambos plates, sino no se puede copiar.
        */
        for(int i = 0; i < this.replicatedNodes.size(); i++){
            List<Node> copies = this.replicatedNodes.get(i);

            for(int j = 0; j < copies.size(); j++) {
                Node copy = copies.get(j);
                Variable copyVariable = copy.getMainVariable();

                if(plateuIIDReplication.replicatedVarsToNode.get(i).containsKey(copyVariable)){
                    Node node = plateuIIDReplication.replicatedVarsToNode.get(i).get(copyVariable);

                    copy.setAssignment(node.getAssignment());
                    copy.setObserved(node.isObserved());
                    copy.setActive(node.isActive());
                    copy.setParallelActivated(node.isParallelActivated());

                    if(!node.isObserved() && !omittedVariables.contains(copyVariable)){
                        for (int k = 0; k < node.getQDist().getNaturalParameters().size(); k++) {
                            copy.getQDist().getNaturalParameters().set(k, node.getQDist().getNaturalParameters().get(k));
                            copy.getQDist().getMomentParameters().set(k, node.getQDist().getMomentParameters().get(k));
                        }
                    }
                }
            }
        }

        /* VMP copy*/
        // MyNote: Muchas de estas asignaciones no son necesarioas, pero lo hago por si acaso
        this.getVMP().setMaxIter(plateuIIDReplication.getVMP().getMaxIter());
        this.getVMP().setSeed(plateuIIDReplication.getVMP().getSeed());
        this.getVMP().setThreshold(plateuIIDReplication.getVMP().getThreshold());
        this.getVMP().setOutput(plateuIIDReplication.getVMP().isOutput());
        this.getVMP().setLocal_elbo(plateuIIDReplication.getVMP().getLocal_elbo());
        this.getVMP().setLocal_iter(plateuIIDReplication.getVMP().getLocal_iter());
        this.getVMP().setParallelMode(plateuIIDReplication.getVMP().isParallelMode());
        this.getVMP().setProbOfEvidence(plateuIIDReplication.getVMP().getLogProbabilityOfEvidence());
        this.getVMP().setnIter(plateuIIDReplication.getVMP().getnIter());

    }

    public PlateuIIDReplication(PlateuIIDReplication plateuIIDReplication, DAG dag) {

        /* initTransientStructure */
        this.replicatedVarsToNode = new ArrayList<>();
        this.nonReplicatedVarsToNode = new ConcurrentHashMap<>();
        this.replicatedNodes = new ArrayList<>();
        this.nonReplicatedNodes = new ArrayList();
        this.nReplications = plateuIIDReplication.nReplications;

        /* Asignamos el DAG y modificamos los campos de las nuevas Parameter variables segun los valores del antiguo Plateau */
        this.setDAG(dag);

        // En principio no es necesario ya que asigna los varIDs correctamente, funciona diferente a Voltric
        /*
        for(int i= 0; i < plateuIIDReplication.nonReplicatedVariablesList.size(); i++){
            Variable parameterVariable = plateuIIDReplication.nonReplicatedVariablesList.get(i);
            Variable copyParameterVariable = this.nonReplicatedVariablesList.get(i);
            copyParameterVariable.setVarID(parameterVariable.getVarID());
            copyParameterVariable.setName(parameterVariable.getName());
            copyParameterVariable.setNumberOfStates(parameterVariable.getNumberOfStates());
            copyParameterVariable.setDistributionTypeEnum(parameterVariable.getDistributionTypeEnum());
            copyParameterVariable.setObservable(parameterVariable.isObservable());
            copyParameterVariable.setStateSpaceType(parameterVariable.getStateSpaceType());
            // El distributionTYpe es el unico que no se puede hacer directamente ya que tiene una referencia interna a la variable
            copyParameterVariable.setDistributionType(copyParameterVariable.getDistributionTypeEnum().newDistributionType(copyParameterVariable));
        }
        */

        this.replicateModel();

        /* Iteramos por los nodos no replicados y copiamos su contenido */
        for(int i = 0; i < plateuIIDReplication.nonReplicatedNodes.size(); i++){
            Node node = plateuIIDReplication.nonReplicatedNodes.get(i);
            Node copy = this.nonReplicatedNodes.get(i);

            for(int j = 0; j < node.getQDist().getNaturalParameters().size(); j++) {
                copy.getQDist().getNaturalParameters().set(j, node.getQDist().getNaturalParameters().get(j));
                copy.getQDist().getMomentParameters().set(j, node.getQDist().getMomentParameters().get(j));

                copy.getPDist().getNaturalParameters().set(j, node.getPDist().getNaturalParameters().get(j));
                copy.getPDist().getMomentParameters().set(j, node.getPDist().getMomentParameters().get(j));
            }
        }

        /* Iteramos por los nodos replicados y copiamos su contenido */
        for(int i = 0; i < plateuIIDReplication.replicatedNodes.size(); i++){
            List<Node> nodes = plateuIIDReplication.replicatedNodes.get(i);
            List<Node> copies = this.replicatedNodes.get(i);

            for(int j = 0; j < nodes.size(); j++){
                Node node = nodes.get(j);
                Node copy = copies.get(j);

                copy.setAssignment(node.getAssignment());
                copy.setObserved(node.isObserved());
                copy.setActive(node.isActive());
                copy.setParallelActivated(node.isParallelActivated());

                if(!node.isObserved()) { // MyNote: Si llamas a la QDist de un nodo observado, te devuelve null
                    for (int k = 0; k < node.getQDist().getNaturalParameters().size(); k++) {
                        copy.getQDist().getNaturalParameters().set(k, node.getQDist().getNaturalParameters().get(k));
                        copy.getQDist().getMomentParameters().set(k, node.getQDist().getMomentParameters().get(k));
                    }
                }
            }
        }

        /* VMP copy */
        // MyNote: Muchas de estas asignaciones no son necesarioas, pero lo hago por si acaso
        this.getVMP().setMaxIter(plateuIIDReplication.getVMP().getMaxIter());
        this.getVMP().setSeed(plateuIIDReplication.getVMP().getSeed());
        this.getVMP().setThreshold(plateuIIDReplication.getVMP().getThreshold());
        this.getVMP().setOutput(plateuIIDReplication.getVMP().isOutput());
        this.getVMP().setLocal_elbo(plateuIIDReplication.getVMP().getLocal_elbo());
        this.getVMP().setLocal_iter(plateuIIDReplication.getVMP().getLocal_iter());
        this.getVMP().setParallelMode(plateuIIDReplication.getVMP().isParallelMode());
        this.getVMP().setProbOfEvidence(plateuIIDReplication.getVMP().getLogProbabilityOfEvidence());
        this.getVMP().setnIter(plateuIIDReplication.getVMP().getnIter());
    }


    /**
     * Sets the evidence for this PlateuStructure.
     *
     * @param data a {@code List} of {@link DataInstance}.
     */
    @Override
    public void setEvidence(List<? extends DataInstance> data) {
        if (data.size() > nReplications)
            throw new IllegalArgumentException("The size of the data is bigger than the number of repetitions");

        for (int i = 0; i < nReplications && i < data.size(); i++) {
            final int slice = i;
            this.replicatedNodes.get(i).forEach(node -> {
                node.setAssignment(data.get(slice));
                node.setActive(true);
            });
        }

        for (int i = data.size(); i < nReplications; i++) {
            this.replicatedNodes.get(i).forEach(node -> {
                node.setAssignment(null);
                node.setActive(false);
            });
        }



        //Non-replicated nodes can have evidende, which is taken from the first data sample in the list
        for (Node nonReplictedNode : this.nonReplicatedNodes) {
            nonReplictedNode.setAssignment(data.get(0));
        }


    }


    /**
     * Replicates this model.
     */
    @Override
    public void replicateModel(){

        nonReplicatedNodes = ef_learningmodel.getDistributionList().values().stream()
                .filter(dist -> isNonReplicatedVar(dist.getVariable()))
                .map(dist -> {
                    Node node = new Node(dist);
                    nonReplicatedVarsToNode.put(dist.getVariable(), node);
                    return node;
                })
                .collect(Collectors.toList());

        for (int i = 0; i < nReplications; i++) {

            Map<Variable, Node> map = new ConcurrentHashMap<>();
            List<Node> tmpNodes = ef_learningmodel.getDistributionList().values().stream()
                    .filter(dist -> isReplicatedVar(dist.getVariable()))
                    .map(dist -> {
                        Node node = new Node(dist);
                        map.put(dist.getVariable(), node);
                        return node;
                    })
                    .collect(Collectors.toList());
            this.replicatedVarsToNode.add(map);
            replicatedNodes.add(tmpNodes);
        }

        for (int i = 0; i < nReplications; i++) {
            for (Node node : replicatedNodes.get(i)) {
                final int slice = i;
                node.setParents(node.getPDist().getConditioningVariables().stream().map(var -> this.getNodeOfVar(var, slice)).collect(Collectors.toList()));
                node.getPDist().getConditioningVariables().stream().forEach(var -> this.getNodeOfVar(var, slice).getChildren().add(node));
            }
        }

        List<Node> allNodes = new ArrayList();


        for (int i = 0; i < nReplications; i++) {
            allNodes.addAll(this.replicatedNodes.get(i));
        }

        allNodes.addAll(this.nonReplicatedNodes);

        this.vmp.setNodes(allNodes);
    }

    @Override
    public PlateuStructure deepCopy(DAG dag) {
        return new PlateuIIDReplication(this, dag);
    }

    @Override
    public PlateuStructure deepCopy(DAG dag, Set<Variable> omittedVariables) {
        return new PlateuIIDReplication(this, dag, omittedVariables);
    }

    private String removeIndexFromName(String name) {

        StringBuilder sb = new StringBuilder();
        boolean addCharacters = false;
        for(int i = name.length() - 1; i >= 0; i-- ){

            char c = name.charAt(i);

            if(c == '_')
                addCharacters = true;

            if(addCharacters)
                sb.append(c);
        }
        return sb.reverse().toString();
    }
}