package eu.amidst.extension.learn.parameter;

import eu.amidst.core.inference.messagepassing.Node;
import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.DAG;
import eu.amidst.core.utils.CompoundVector;
import eu.amidst.core.variables.Variable;
import eu.amidst.extension.learn.parameter.penalizer.NoPenalizer;
import eu.amidst.extension.util.GraphUtilsAmidst;
import eu.amidst.extension.util.tuple.Tuple2;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Version local del VBEM. Se suele llamar para aprender partes del modelo sin tener en cuenta su conjunto.
 *
 * Ofrece la posibilidad de penalizar el score. Dicha penalizacion se aplica al final ya que es un valor constante durante
 * el aprendizaje de parametros.
 */
public class VBEM_Local {

    /** Represents the plateu structure {@link PlateuStructure}*/
    protected PlateuStructure plateuStructure;

    protected VBEMConfig config;

    public VBEM_Local() {
        InitializationVBEM initializationVBEM = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, false);
        this.config = new VBEMConfig(0,0.01, 100, initializationVBEM, new NoPenalizer());
    }

    public VBEM_Local(VBEMConfig config) {
        this.config = config;
    }
    /**
     * Returns the associated PlateauStructure object
     * @return
     */
    public PlateuStructure getPlateuStructure() { return this.plateuStructure; }

/*
    public double learnModel(PlateuStructure model, DAG dag, Variable variable) {
        Set<Variable> variables = new LinkedHashSet<>();
        variables.add(variable);
        return learnModel(model, dag, variables);
    }
*/
    public double learnModel(PlateuStructure model, DAG dag, Set<Variable> variables) {
        List<Node> localNodes = initialization(model, dag, variables);

        this.plateuStructure.emInference(localNodes);

        double penalizedElbo = this.config.elboPenalizer().penalize(this.plateuStructure.getLogProbabilityOfEvidence(), dag);
        this.getPlateuStructure().getVMP().setProbOfEvidence(penalizedElbo);

        return penalizedElbo;
    }

    /* MyNote: No se llama a setEvidence porque al igual que con VBEM_HC, asumimos que ya ha sido asignada de antemano con una iteracion VBEM */
    private List<Node> initialization(PlateuStructure model, DAG dag, Set<Variable> variables) {
        this.plateuStructure = model;
        this.plateuStructure.getVMP().setThreshold(this.config.threshold());
        this.plateuStructure.getVMP().setMaxIter(this.config.maxIterations());
        this.plateuStructure.getVMP().setOutput(false);
        this.plateuStructure.getVMP().setTestELBO(false);
        this.plateuStructure.setSeed(this.config.seed());

        //List<Node> localNodes = computeLocalNodes(variables, dag);
        List<Node> localNodes = computeLocalNodes(variables);

        switch (this.config.initializationTypeVBEM()) {
            case RANDOM:
                randomInitialization(
                        localNodes,
                        this.config.initCandidates(),
                        this.config.initIterations(),
                        this.config.initTestConvergence());
                break;
            case PYRAMID:
                pyramidInitialization(
                        localNodes,
                        this.config.initCandidates(),
                        this.config.initIterations(),
                        this.config.initTestConvergence());
            case NONE:
                break;
        }

        return localNodes;
    }


    private List<Node> computeLocalNodes(Set<Variable> variables) {
        List<Node> localNodes = new ArrayList<>();
        for(Variable variable: variables)
            localNodes.addAll(this.plateuStructure.getLatentNodes(variable));
        return localNodes;
    }

    private List<Node> computeLocalNodes(Set<Variable> variables, DAG dag) {

        /* Primero seleccionamos todas las variables no repetidas que intervienen (las variables latentes en cuestion y sus hijos) */
        Set<Variable> variablesToConsider = new LinkedHashSet<>();
        for(Variable variable: variables) {
            variablesToConsider.add(variable);
            variablesToConsider.addAll(GraphUtilsAmidst.getChildren(variable, dag));
        }

        /* Después seleccionamos los nodos latentes de dichas variables */
        List<Node> localNodes = new ArrayList<>();
        for(Variable variable: variablesToConsider)
            localNodes.addAll(this.plateuStructure.getLatentNodes(variable));

        return localNodes;
    }
    private void randomInitialization(List<Node> localNodes,
                                      int randomRestarts,
                                      int randomIterations,
                                      boolean withConvergence) {

        CompoundVector bestPosterior = this.plateuStructure.getLatentVariablesPosterior(localNodes);
        double bestScore = -Double.MAX_VALUE;

        for(int i=0; i < randomRestarts; i++) {
            this.plateuStructure.resetQs(localNodes);

            if(withConvergence)
                this.plateuStructure.emInference(localNodes, randomIterations);
            else
                this.plateuStructure.emInferenceWithoutConvergence(localNodes, randomIterations);

            double score = this.plateuStructure.getLogProbabilityOfEvidence();

            if(score > bestScore) {
                bestScore = score;
                bestPosterior = this.plateuStructure.getLatentVariablesPosterior(localNodes);
            }
        }

        this.plateuStructure.updateLatentVariablesPosterior(localNodes, bestPosterior);
    }

    /**
     * First we sample n initial configurations of the parameters. Next we perform one VBEM step and retain n/2 of the
     * configurations that led to largest values of score. Then we perform two VBEM steps and retain n/4 configurations.
     * We continue this procedure, doubling the number of VBEM steps at each iteration until only one configuration remain.
     * @param initialNumberOfCandidates
     * @param maxIterations
     */
    private void pyramidInitialization(List<Node> localNodes, int initialNumberOfCandidates, int maxIterations, boolean withConvergence) {

        int currentIterations = 1;
        List<Tuple2<CompoundVector, Double>> candidates = new ArrayList<>(initialNumberOfCandidates);

        /* Generate the candidates by random generation of parameters and 1 run of VBEM */
        for(int i = 0; i < initialNumberOfCandidates; i++) {
            this.plateuStructure.resetQs(localNodes);
            if(withConvergence)
                this.plateuStructure.emInference(localNodes, currentIterations);
            else
                this.plateuStructure.emInferenceWithoutConvergence(localNodes, currentIterations);
            candidates.add(new Tuple2<>(this.plateuStructure.getLatentVariablesPosterior(localNodes), this.plateuStructure.getLogProbabilityOfEvidence()));
        }
        /* Pyramidal iteration */
        while(candidates.size() > 1) {

            /* Sort candidates by score value */
            candidates.sort(new PyramidCandidateComparator<>());

            /* Remove (currentNumberOfCandidates / 2) candidates */
            int candidatesSize = candidates.size();
            for(int i = candidatesSize - 1; i >= (candidatesSize/2); i--)
                candidates.remove(i);

            /* Run VBEM on each candidate */
            currentIterations = currentIterations * 2;
            if(currentIterations > maxIterations)
                currentIterations = maxIterations;

            List<Tuple2<CompoundVector, Double>> auxListOfCandidates = new ArrayList<>(candidates.size());
            auxListOfCandidates.addAll(candidates);
            candidates = new ArrayList<>(auxListOfCandidates.size());

            for(int i = 0; i < auxListOfCandidates.size(); i++) {
                this.plateuStructure.updateLatentVariablesPosterior(localNodes, auxListOfCandidates.get(i).getFirst());
                this.plateuStructure.getVMP().setLocal_elbo(auxListOfCandidates.get(i).getSecond());
                this.plateuStructure.getVMP().setProbOfEvidence(auxListOfCandidates.get(i).getSecond());
                if(withConvergence)
                    this.plateuStructure.emInference(localNodes, currentIterations);
                else
                    this.plateuStructure.emInferenceWithoutConvergence(localNodes, currentIterations);
                candidates.add(new Tuple2<>(this.plateuStructure.getLatentVariablesPosterior(localNodes), this.plateuStructure.getLogProbabilityOfEvidence()));
            }
        }
    }
}
