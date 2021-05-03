package eu.amidst.extension.learn.parameter;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.exponentialfamily.EF_LearningBayesianNetwork;
import eu.amidst.core.learning.parametric.bayesian.utils.PlateuIIDReplication;
import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.utils.CompoundVector;
import eu.amidst.extension.learn.parameter.penalizer.NoPenalizer;
import eu.amidst.extension.util.tuple.Tuple2;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/*
    La idea de esta clase va a ser copiar gran parte de SVB y adaptarla para que sirva para datos en memoria. Incluir aqui
    todas las mejoras que necesitamos ponerle al SVB pero sin modificar a ese, y ademas nos sirve para comparar el uno con
    el otro no sea que nos hayamos equivocado.

    2 metodos principales:
        - learnModelWithPriorUpdate -> Combina el antiguo updateModel con initLearning y genera un modelo que se almacena en la clase
        - updateModel -> Elimina la parte de initLearning y simplemente actualiza el modelo con nuevos datos

    Dado que EF_Learning_BN no tiene un grafo asociado, debemos guardarlos por separado (la razon es porque contiene parameter vars,
    pero en una buena libreria se podrian filtrar y punto)

    TODO: Estudiar como calcular el ELBO para un nuevo dataSet con VMP (modelo ya aprendido)
 */
// TODO: Revision 22-09-2020 -> Creo que la inicializacion piramidal no esta bien ya que no devuelvo el mejor candidato,
// aunque si exploro el espacio de candidatos correctamente.
public class VBEM {

    /** Reference to the PlateauStructure's EF_LearningBayesianNetwork object. It represents the learned network */
    protected EF_LearningBayesianNetwork ef_extendedBN;

    /** Represents a directed acyclic graph {@link DAG}. */
    protected DAG dag;

    /** Represents the plateu structure {@link PlateuStructure}*/
    protected PlateuStructure plateuStructure = new PlateuIIDReplication();

    protected VBEMConfig config;

    public VBEM() {
        InitializationVBEM initializationVBEM = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, false);
        this.config = new VBEMConfig(0,0.01, 100, initializationVBEM, new NoPenalizer());
    }

    public VBEM(VBEMConfig config) {
        this.config = config;
    }

    public PlateuStructure getPlateuStructure() {
        return this.plateuStructure;
    }

    public BayesianNetwork getLearntBayesianNetwork() {
        return new BayesianNetwork(this.dag, this.ef_extendedBN.toConditionalDistribution());
    }

    /** */
    public double learnModelWithPriorUpdate(DataOnMemory<DataInstance> data, DAG dag, Map<String, double[]> priorsParameters) {
        initialization(data, dag, priorsParameters);

        this.plateuStructure.emInference();
        double penalizedElbo = this.config.elboPenalizer().penalize(this.plateuStructure.getLogProbabilityOfEvidence(), dag);
        this.getPlateuStructure().getVMP().setProbOfEvidence(penalizedElbo);

        this.plateuStructure.updateParameterVariablesPrior(this.plateuStructure.getParameterVariablesPosterior());

        return this.plateuStructure.getLogProbabilityOfEvidence();
    }

    /** */
    public double learnModelWithPriorUpdate(DataOnMemory<DataInstance> data, DAG dag) {
        return learnModelWithPriorUpdate(data, dag, new HashMap<>());
    }

    /** */
    public double learnModel(DataOnMemory<DataInstance> data, DAG dag, Map<String, double[]> priorsParameters) {
        initialization(data, dag, priorsParameters);

        this.plateuStructure.emInference();
        double penalizedElbo = this.config.elboPenalizer().penalize(this.plateuStructure.getLogProbabilityOfEvidence(), dag);
        this.getPlateuStructure().getVMP().setProbOfEvidence(penalizedElbo);

        return this.plateuStructure.getLogProbabilityOfEvidence();
    }

    /** */
    public double learnModel(DataOnMemory<DataInstance> data, DAG dag) {
        return learnModel(data, dag, new HashMap<>());
    }

    /*
        MyNote: Se corresponderia con un update Bayesiano, es decir, que llega nueva información en forma de batch de instancias
        que no cambia lo anterior pero que si cambia los valores esperados.

        Tal y como sabemos, una vez se actualiza el modelo con un batch de datos, es necesario cambiar la prior por la
        posterior resultante, de tal forma que cuando llegue un nuevo batch no se empiece de 0, ya que no tenemos una prior
        "vacia", sino la resultante del aprendizaje

        Es tambien necesario añadir una nueva repeticion al plateu
     */
    public double updateModel(DataOnMemory<DataInstance> batch) {
        return 0;
    }

    private void initialization(DataOnMemory<DataInstance> data, DAG dag, Map<String, double[]> priorsParameters) {

        this.plateuStructure.initTransientDataStructure();
        this.plateuStructure.setNRepetitions(data.getNumberOfDataInstances());
        this.plateuStructure.setSeed(this.config.seed());
        this.plateuStructure.setDAG(dag, priorsParameters);
        this.plateuStructure.replicateModel();
        this.plateuStructure.getVMP().setOutput(false);
        this.plateuStructure.getVMP().setTestELBO(false);
        this.plateuStructure.getVMP().setThreshold(this.config.threshold());
        this.plateuStructure.getVMP().setMaxIter(this.config.maxIterations());

        switch (this.config.initializationTypeVBEM()) {
            case RANDOM: randomInitialization(data); break;
            case PYRAMID: pyramidInitialization(data); break;
        }
        
        this.dag = dag;
        this.ef_extendedBN = this.plateuStructure.getEFLearningBN();
    }


    private void randomInitialization(DataOnMemory<DataInstance> data) {

        CompoundVector bestPosterior = null;
        double bestScore = -Double.MAX_VALUE;

        this.plateuStructure.setEvidence(data.getList());
        for(int i = 0; i < config.initCandidates(); i++) {
            this.plateuStructure.resetQs();

            if(this.config.initTestConvergence())
                this.plateuStructure.emInference(this.config.initIterations());
            else
                this.plateuStructure.emInferenceWithoutConvergence(config.initIterations());

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
     */
    private void pyramidInitialization(DataOnMemory<DataInstance> data) {

        int currentIterations = 1;
        List<Tuple2<CompoundVector, Double>> candidates = new ArrayList<>(config.initCandidates());
        candidates.add(new Tuple2<>(this.plateuStructure.getLatentVariablesPosterior(), this.plateuStructure.getLogProbabilityOfEvidence()));
        this.plateuStructure.setEvidence(data.getList());

        /* Generate the candidates by random generation of parameters and 1 run of VBEM */
        for(int i = candidates.size(); i < this.config.initCandidates(); i++) {
            this.plateuStructure.resetQs();
            if(config.initTestConvergence())
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
            for(int i = candidates.size() - 1; i >= (candidates.size()/2); i--)
                candidates.remove(i);

            /* Run VBEM on each candidate */
            currentIterations = currentIterations * 2;
            if(currentIterations > this.config.initIterations())
                currentIterations = this.config.initIterations();

            List<Tuple2<CompoundVector, Double>> auxListOfCandidates = new ArrayList<>(candidates.size());
            auxListOfCandidates.addAll(candidates);
            candidates = new ArrayList<>(auxListOfCandidates.size());

            for(int i = 0; i < auxListOfCandidates.size(); i++) {
                this.plateuStructure.updateLatentVariablesPosterior(auxListOfCandidates.get(i).getFirst());
                if(config.initTestConvergence())
                    this.plateuStructure.emInference(currentIterations);
                else
                    this.plateuStructure.emInferenceWithoutConvergence(currentIterations);
                candidates.add(new Tuple2<>(this.plateuStructure.getLatentVariablesPosterior(), this.plateuStructure.getLogProbabilityOfEvidence()));
            }
        }
    }
}
