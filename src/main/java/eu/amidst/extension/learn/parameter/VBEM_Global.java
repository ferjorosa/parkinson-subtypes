package eu.amidst.extension.learn.parameter;

import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.DAG;
import eu.amidst.core.utils.CompoundVector;
import eu.amidst.extension.learn.parameter.penalizer.NoPenalizer;
import eu.amidst.extension.util.tuple.Tuple2;

import java.util.ArrayList;
import java.util.List;

/**
 * Este es la version full del VBEM que se llama al escoger un modelo dentro de un BltmHcOperator.
 *
 * Se distingue de VBEM_HC en los parametros de learnModelWithPriorUpdate y en la inicializacion. Se asume que la evidencia ya ha sido
 * asignada, se asume que se ha realizado algun tipo de preparacion o aprendizaje previo. Por ejemplo que se haya llamado
 * antes a VBEM_HC
 *
 * - Se distingue de VBEM_Local en que al igual que VBEM_HC no omite variables en el aprendizaje
 * - Ademas, ofrece la posibilidad de penalizar el score
 */
public class VBEM_Global {

    /** Represents the plateu structure {@link PlateuStructure}*/
    protected PlateuStructure plateuStructure;

    protected VBEMConfig config;

    public VBEM_Global() {
        InitializationVBEM initializationVBEM = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, false);
        this.config = new VBEMConfig(0,0.01, 100, initializationVBEM, new NoPenalizer());
    }

    public VBEM_Global(VBEMConfig config) {
        this.config = config;
    }

    /**
     * Returns the associated PlateauStructure object
     * @return
     */
    public PlateuStructure getPlateuStructure() { return this.plateuStructure; }

    public double learnModel(PlateuStructure model, DAG dag) {

        initialization(model);

        this.plateuStructure.emInference();
        double penalizedElbo = this.config.elboPenalizer().penalize(this.plateuStructure.getLogProbabilityOfEvidence(), dag);
        this.getPlateuStructure().getVMP().setProbOfEvidence(penalizedElbo);

        return penalizedElbo;
    }

    /* MyNote: No se llama a setEvidence porque al igual que con VBEM_Local, asumimos que ya ha sido asignada de antemano con una iteracion VBEM */
    private void initialization(PlateuStructure model) {
        this.plateuStructure = model;
        this.plateuStructure.getVMP().setThreshold(this.config.threshold());
        this.plateuStructure.getVMP().setMaxIter(this.config.maxIterations());
        this.plateuStructure.getVMP().setOutput(false);
        this.plateuStructure.getVMP().setTestELBO(false);
        this.plateuStructure.setSeed(this.config.seed());

        switch (this.config.initializationTypeVBEM()) {
            case RANDOM:
                randomInitialization(
                    this.config.initCandidates(),
                    this.config.initIterations(),
                    this.config.initTestConvergence());
                break;
            case PYRAMID:
                pyramidInitialization(
                        this.config.initCandidates(),
                        this.config.initIterations(),
                        this.config.initTestConvergence());
            case NONE:
                break;
        }
    }

    private void randomInitialization(int randomRestarts, int randomIterations, boolean withConvergence) {

        CompoundVector bestPosterior = this.plateuStructure.getLatentVariablesPosterior();
        double bestScore = this.plateuStructure.getLogProbabilityOfEvidence();

        for(int i=0; i < randomRestarts; i++) {
            this.plateuStructure.resetQs();

            if(withConvergence)
                this.plateuStructure.emInference(randomIterations);
            else
                this.plateuStructure.emInferenceWithoutConvergence(randomIterations);

            double score = this.plateuStructure.getLogProbabilityOfEvidence();

            if(score > bestScore) {
                bestScore = score;
                bestPosterior = this.plateuStructure.getLatentVariablesPosterior();
            }
        }

        this.plateuStructure.updateLatentVariablesPosterior(bestPosterior);
    }

    /**
     * First we sample n initial configurations of the parameters. Next we perform one VBEM step and retain n/2 of the
     * configurations that led to largest values of score. Then we perform two VBEM steps and retain n/4 configurations.
     * We continue this procedure, doubling the number of VBEM steps at each iteration until only one configuration remain.
     * @param initialNumberOfCandidates
     * @param maxIterations
     */
    private void pyramidInitialization(int initialNumberOfCandidates, int maxIterations, boolean withConvergence) {

        int currentIterations = 1;
        List<Tuple2<CompoundVector, Double>> candidates = new ArrayList<>(initialNumberOfCandidates);
        candidates.add(new Tuple2<>(this.plateuStructure.getLatentVariablesPosterior(), this.plateuStructure.getLogProbabilityOfEvidence()));

        /* Generate the candidates by random generation of parameters and 1 run of VBEM */
        for(int i = candidates.size(); i < initialNumberOfCandidates; i++) {
            this.plateuStructure.resetQs();
            if(withConvergence)
                this.plateuStructure.emInference(currentIterations);
            else
                this.plateuStructure.emInferenceWithoutConvergence(currentIterations);
            candidates.add(new Tuple2<>(this.plateuStructure.getLatentVariablesPosterior(), this.plateuStructure.getLogProbabilityOfEvidence()));
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
                this.plateuStructure.updateLatentVariablesPosterior(auxListOfCandidates.get(i).getFirst());
                this.plateuStructure.getVMP().setLocal_elbo(auxListOfCandidates.get(i).getSecond());
                this.plateuStructure.getVMP().setProbOfEvidence(auxListOfCandidates.get(i).getSecond());
                if(withConvergence)
                    this.plateuStructure.emInference(currentIterations);
                else
                    this.plateuStructure.emInferenceWithoutConvergence(currentIterations);
                candidates.add(new Tuple2<>(this.plateuStructure.getLatentVariablesPosterior(), this.plateuStructure.getLogProbabilityOfEvidence()));
            }
        }
    }
}
