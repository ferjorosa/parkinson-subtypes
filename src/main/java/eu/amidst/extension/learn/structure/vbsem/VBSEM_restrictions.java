package eu.amidst.extension.learn.structure.vbsem;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.extension.data.DataUtils;
import eu.amidst.extension.learn.parameter.VBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHc;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHcConfig;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHcResult;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcOperator;
import eu.amidst.extension.util.DagSampler;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.PriorsFromData;
import eu.amidst.extension.util.tuple.Tuple2;
import eu.amidst.extension.util.tuple.Tuple3;

import java.util.*;
import java.util.stream.Collectors;

// TODO: Por ahora las priors son solo con Empirical Bayes, podemos expandirlo introduciendo priors especificas
// TODO: Por ahora las priors se ponen sin contar variables latentes (esto incluye missing)

// Todo: 24-11-2020: DagSampler espera un
public class VBSEM_restrictions {

    private VBEMConfig config;

    private BayesianHcConfig bayesianHcConfig;

    private InitializationVBSEM initializationVBSEM;

    private int maxIterations;

    public VBSEM_restrictions(VBEMConfig config,
                              BayesianHcConfig bayesianHcConfig,
                              InitializationVBSEM initializationVBSEM,
                              int maxIterations) {
        this.config = config;
        this.bayesianHcConfig = bayesianHcConfig;
        this.initializationVBSEM = initializationVBSEM;
        this.maxIterations = maxIterations;
    }

    public Tuple2<BayesianNetwork, Double> learnModel(BayesianNetwork baseModel,
                                                      double baseScore,
                                                      DataOnMemory<DataInstance> data,
                                                      Set<BayesianHcOperator> hcOperators,
                                                      LogUtils.LogLevel vbsemLogLevel,
                                                      LogUtils.LogLevel hcLogLevel) {

        BayesianHc bayesianHc = new BayesianHc(this.bayesianHcConfig, this.bayesianHcConfig.getMaxIter(), this.bayesianHcConfig.getThreshold(), hcOperators);
        VBEM vbem = new VBEM(config);

        LogUtils.info("\nVBSEM - Base score: "+baseScore, vbsemLogLevel);

        Tuple2<BayesianNetwork, Double> currentResult = initialization(baseModel, baseScore, data, bayesianHc, vbem, vbsemLogLevel, hcLogLevel);
        int iterations = 1;

        while(iterations <= maxIterations) {
            iterations++;
            Tuple2<BayesianNetwork, Double> newResult = vbsemStep(currentResult.getFirst(), data, bayesianHc, vbem, hcLogLevel);
            double newScore = newResult.getSecond();

            /* Update the current model if the score has increased. If not, return the current model */
            if(newScore <= currentResult.getSecond() || Math.abs(newScore - currentResult.getSecond()) < config.threshold()) {
                LogUtils.info("VBSEM - Failed to improve model: "+ newScore, vbsemLogLevel);

                setLatentVariables(currentResult.getFirst(), data); // Set latent variables as not observable

                return currentResult;
            }

            LogUtils.info("VBSEM - Model updated: " + newScore, vbsemLogLevel);
            currentResult = newResult;
        }

       setLatentVariables(currentResult.getFirst(), data); // Set latent variables as not observable

        return currentResult;
    }

    public Tuple2<BayesianNetwork, Double> vbsemStep(BayesianNetwork model,
                                                     DataOnMemory<DataInstance> data,
                                                     BayesianHc bayesianHc,
                                                     VBEM vbem,
                                                     LogUtils.LogLevel hcLogLevel) {

        /* Generate complete data using currentModel */
        DataOnMemory<DataInstance> completeData = DataUtils.completeLatentData(data, model);

        /* Assign the new attributes to the latent variables in the current model */
        List<String> latentVarNames = new ArrayList<>();
        for(Variable variable: model.getVariables()) {
            if (variable.getAttribute() == null) {
                latentVarNames.add(variable.getName());
                variable.setAttribute(completeData.getAttributes().getAttributeByName(variable.getName()));
                variable.setObservable(true);
            }
        }

        /* Learn a new structure S[n+1] using the complete data (and Empirical Bayes priors) */
        final Map<String, double[]> priors = PriorsFromData.generate(data, 1);
        BayesianHcResult bayesianHcResult = bayesianHc.learnModel(model.getDAG(), completeData, priors, hcLogLevel);

        /* Before VBEM, remove attributes of latent variables and set them as no observable */
        for(String latentVarName: latentVarNames){
            Variable latentVar = bayesianHcResult.getDag().getVariables().getVariableByName(latentVarName);
            latentVar.setObservable(false);
            latentVar.setAttribute(null);
        }

        /* Learn the true parameters Theta[n+1] of the structure S[n+1] */
        double newScore = vbem.learnModelWithPriorUpdate(data, bayesianHcResult.getDag(), priors);
        BayesianNetwork newModel = vbem.getLearntBayesianNetwork();

        return new Tuple2<>(newModel, newScore);
    }

    /** It is only internally used in the Pyramid initialization, that is why it doesn't have proper Logging */
    private Tuple3<BayesianNetwork, Double, Boolean> vbsemSteps(int nIterations,
                                                                BayesianNetwork baseModel,
                                                                double baseScore,
                                                                DataOnMemory<DataInstance> data,
                                                                BayesianHc bayesianHc,
                                                                VBEM vbem,
                                                                LogUtils.LogLevel hcLogLevel) {

        /* Do an initial VBSEM step to start the process */
        int iterations = 1;
        Tuple2<BayesianNetwork, Double> currentResult = vbsemStep(baseModel, data, bayesianHc, vbem, hcLogLevel);
        iterations++;

        if(currentResult.getSecond() <= baseScore || Math.abs(currentResult.getSecond() - baseScore) < config.threshold())
            return new Tuple3<>(baseModel, baseScore, true); // Early convergence

        while(iterations <= nIterations) {
            iterations++;
            Tuple2<BayesianNetwork, Double> newResult = vbsemStep(currentResult.getFirst(), data, bayesianHc, vbem, hcLogLevel);
            double newScore = newResult.getSecond();

            /* Update the current model if the score has increased. If not, return the current model */
            if(newScore <= currentResult.getSecond() || Math.abs(newScore - currentResult.getSecond()) < config.threshold()) {
                return new Tuple3<>(currentResult.getFirst(), currentResult.getSecond(), true); // Early convergence
            }

            currentResult = newResult;
        }

        return new Tuple3<>(currentResult.getFirst(), currentResult.getSecond(), true); // Didnt converge after all the steps
    }

    private Tuple2<BayesianNetwork, Double> initialization(BayesianNetwork baseModel,
                                                           double baseScore,
                                                           DataOnMemory<DataInstance> data,
                                                           BayesianHc bayesianHc,
                                                           VBEM vbem,
                                                           LogUtils.LogLevel vbsemLogLevel,
                                                           LogUtils.LogLevel hcLogLevel) {
        if(this.initializationVBSEM.initializationType() == InitializationTypeVBSEM.NONE)
                return new Tuple2<>(baseModel, baseScore);

        else if(this.initializationVBSEM.initializationType() == InitializationTypeVBSEM.PYRAMID) {

            throw new IllegalStateException("DagSampler y maxNumberOfParents no estan actualizados porque ahora podemos hacer que una variable tenga un " +
                    "numero de padres especifico, lo que antes era un valor por defecto para todos");

            // TODO: 24-11-2020:  Es necesario actualizar Dag Sampler para que permita un numero maximo de padres diferente para cada variable
//            Set<BayesianHcOperator> operators = bayesianHc.getOperators();
//            Map<Variable, List<Variable>> arcBlackList = new HashMap<>();
//            for(BayesianHcOperator operator: operators)
//                if(operator instanceof BayesianHcAddArc){
//                    BayesianHcAddArc addArcOperator = (BayesianHcAddArc) operator;
//                    arcBlackList = addArcOperator.getArcBlackList();
//                }
//            return pyramidInitialization(baseModel, baseScore, data, bayesianHc, vbem, maxParents, arcBlackList, vbsemLogLevel, hcLogLevel);

        } else
            throw new IllegalArgumentException("Invalid VBSEM initialization");
    }

    /**
     * First we sample n initial configurations of the network. Next we perform one VBSEM step and retain n/2 of the
     * configurations that led to largest values of score. Then we perform two VBSEM steps and retain n/4 configurations.
     * We continue this procedure, doubling the number of VBSEM steps at each iteration until only one configuration remain.
     */
    private Tuple2<BayesianNetwork, Double> pyramidInitialization(BayesianNetwork baseModel,
                                                                  double baseScore,
                                                                  DataOnMemory<DataInstance> data,
                                                                  BayesianHc bayesianHc,
                                                                  VBEM vbem,
                                                                  int maxParents,
                                                                  Map<Variable, List<Variable>> arcBlackList,
                                                                  LogUtils.LogLevel vbsemLogLevel,
                                                                  LogUtils.LogLevel hcLogLevel) {

        LogUtils.debug("\nPyramid initialization with " + this.initializationVBSEM.nCandidates() + "candidates", vbsemLogLevel);

        final Map<String, double[]> priors = PriorsFromData.generate(data, 1);

        /* Initialize the list of candidates with the base model. Each candidate includes:
        * - The model
        * - The score
        * - A boolean variable indicating if the candidate has converged or not
        */
        List<Tuple3<BayesianNetwork, Double, Boolean>> candidates = new ArrayList<>(this.initializationVBSEM.nCandidates());
        candidates.add(new Tuple3<>(baseModel, baseScore, false));

        /* Prepare the arcBlackList (excludedArcs) for DagSampler */
        Map<Integer, Set<Integer>> excludedArcs = new HashMap<>(arcBlackList.size());
        for(int varIndex = 0; varIndex < baseModel.getVariables().getNumberOfVars(); varIndex++) {
            Variable var = baseModel.getVariables().getListOfVariables().get(varIndex);
            List<Variable> excludedChildren = new ArrayList<>();
            if(arcBlackList.containsKey(var))
                excludedChildren = arcBlackList.get(var);
            Set<Integer> excludedChildrenIndexes = excludedChildren.stream()
                    .map(x-> baseModel.getVariables().getListOfVariables().indexOf(x)).collect(Collectors.toSet());
            excludedArcs.put(varIndex, excludedChildrenIndexes);
        }

        /* Generate the candidates by random generation of structure and 1 run of VBSEM */
        DagSampler dagSampler = new DagSampler(config.seed());
        int nSamples = this.initializationVBSEM.nCandidates() - 1; // The base model is one of the candidates
        List<DAG> samples = dagSampler.sample(baseModel.getDAG(), nSamples, maxParents, this.initializationVBSEM.sparsityCoefficient(), excludedArcs);

        /*
            Transform each DAG sample into a candidate by learning its parameters using VBEM, then do a single
            VBSEM step to refine the structure and parameters of each candidate
        */
        int nSteps = 1;
        for(int i = 0; i < nSamples; i++) {
            DAG dag = samples.get(i);
            double score = vbem.learnModelWithPriorUpdate(data, dag, priors);
            BayesianNetwork model = vbem.getLearntBayesianNetwork();
            candidates.add(vbsemSteps(nSteps, model, score, data, bayesianHc, vbem, hcLogLevel));
        }

        /* Pyramidal iteration */
        while (candidates.size() > 1) {

            LogUtils.debug("\nVBSEM candidates ("+candidates.size() + "): ", vbsemLogLevel);
            for (Tuple3<BayesianNetwork, Double, Boolean> candidate: candidates)
                LogUtils.debug("score = " + candidate.getSecond() + ", convergence = " + candidate.getThird(), vbsemLogLevel);

            /* Sort candidates by score value */
            candidates.sort(new PyramidCandidateComparator3());

            /* Remove (currentNumberOfCandidates / 2) candidates */
            for(int i = candidates.size() - 1; i >= (candidates.size()/2); i--)
                candidates.remove(i);

            /* Run VBSEM on each candidate a number of nSteps */
            nSteps = nSteps * 2;
            List<Tuple3<BayesianNetwork, Double, Boolean>> auxListOfCandidates = new ArrayList<>(candidates.size());
            auxListOfCandidates.addAll(candidates);
            candidates = new ArrayList<>(auxListOfCandidates.size());

            for(int i = 0; i < auxListOfCandidates.size(); i++) {
                Tuple3<BayesianNetwork, Double, Boolean> candidate = auxListOfCandidates.get(i);
                if(!candidate.getThird()) { // If the candidate hasn't converged yet, do more VBSEM iterations
                    Tuple3<BayesianNetwork, Double, Boolean> newCandidate = vbsemSteps(nSteps, candidate.getFirst(), candidate.getSecond(), data, bayesianHc, vbem, hcLogLevel);
                    candidates.add(newCandidate);
                } else
                    candidates.add(candidate);
            }
        }

        Tuple3<BayesianNetwork, Double, Boolean> bestCandidate = candidates.get(0);

        return new Tuple2<>(bestCandidate.getFirst(), bestCandidate.getSecond());
    }

    private void setLatentVariables(BayesianNetwork model,
                                    DataOnMemory<DataInstance>data) {
        List<String> attributeNames = data.getAttributes().getFullListOfAttributes().stream().map(Attribute::getName).collect(Collectors.toList());

        for(Variable var: model.getVariables()) {
            if(!attributeNames.contains(var.getName())) {
                var.setAttribute(null);
                var.setObservable(false);
            }
        }
    }
}
