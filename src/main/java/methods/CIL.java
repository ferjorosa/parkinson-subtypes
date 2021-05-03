package methods;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.extension.learn.parameter.InitializationTypeVBEM;
import eu.amidst.extension.learn.parameter.InitializationVBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.penalizer.BishopPenalizer;
import eu.amidst.extension.learn.structure.BLFM_IncLearner_Missing;
import eu.amidst.extension.learn.structure.Result;
import eu.amidst.extension.learn.structure.operator.incremental.BlfmIncAddArc;
import eu.amidst.extension.learn.structure.operator.incremental.BlfmIncAddDiscreteNode;
import eu.amidst.extension.learn.structure.operator.incremental.BlfmIncOperator;
import eu.amidst.extension.learn.structure.typelocalvbem.SimpleLocalVBEM;
import eu.amidst.extension.learn.structure.typelocalvbem.TypeLocalVBEM;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.distance.ChebyshevDistance;
import eu.amidst.extension.util.distance.DistanceFunction;
import eu.amidst.extension.util.tuple.Tuple2;

import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

/**
 * Template class of the constrained incremental learner (CIL) algorithm, which was published in
 * F. Rodriguez-Sanchez, P. Larrañaga and C. Bielza, "Incremental Learning of Latent Forests," in IEEE Access, 2020.
 * and its currently implemented with the name of BLFM_IncLearner in our extension of the AMIDST toolbox.
 *
 * In this template, we define three main variants of CIL:
 *  - High flexibility: it allows arcs from observed variables to other observed/latent variables.
 *  - Medium flexibility: it allows arcs from observed variables to other observed variables.
 *  - Low flexibility: that only allows arcs from latent to other latent or observed variables.
 *
 * In the original article we used the most flexible approach with it is perfect for density estimation. However,
 * when dealing with a clustering problem it may be more interesting to have an easier to understand model that
 * also contains a higher number of discrete latent variables (clustering variables), thus the least flexible model.
 *
 * The idea is to use these three variants to generate different models and analyze how a decrease in flexibility
 * affects how well the model is able to represent the underlying data distribution.
 */
public class CIL {

    public enum Flexibility {
        HIGH,
        MEDIUM,
        LOW
    }

    private BLFM_IncLearner_Missing incLearner;

    public CIL(Flexibility flexibility,
               long seed,
               int nVbemCandidates) {

        InitializationVBEM initialVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 1, 1, false);
        VBEMConfig initialVBEMConfig = new VBEMConfig(seed, 0.01, 100, initialVBEMinitialization, new BishopPenalizer());
        InitializationVBEM localVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 4, 2, true);
        VBEMConfig localVBEMConfig = new VBEMConfig(seed, 0.01, 100, localVBEMinitialization, new BishopPenalizer());
        InitializationVBEM iterationVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 16, 4, true);
        VBEMConfig iterationVBEMConfig = new VBEMConfig(seed, 0.01, 100, iterationVBEMinitialization, new BishopPenalizer());
        InitializationVBEM finalVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, nVbemCandidates, 16, true);
        VBEMConfig finalVBEMConfig = new VBEMConfig(seed, 0.01, 100, finalVBEMinitialization, new BishopPenalizer());

        TypeLocalVBEM typeLocalVBEM = new SimpleLocalVBEM();
        boolean iterationGlobalVBEM = true;

        DistanceFunction distanceFunction_mi = new ChebyshevDistance();
        int n_neighbors_mi = 3;
        boolean gaussian_noise_mi = false;
        boolean normalized_mi = false;

        switch (flexibility) {

            case HIGH: initializeHigh(iterationGlobalVBEM, n_neighbors_mi, distanceFunction_mi, gaussian_noise_mi, seed,
                    normalized_mi, initialVBEMConfig, localVBEMConfig, iterationVBEMConfig, finalVBEMConfig, typeLocalVBEM);
                break;

            case MEDIUM: initializeMedium(iterationGlobalVBEM, n_neighbors_mi, distanceFunction_mi, gaussian_noise_mi, seed,
                    normalized_mi, initialVBEMConfig, localVBEMConfig, iterationVBEMConfig, finalVBEMConfig, typeLocalVBEM);
                break;

            case LOW: initializeLow(iterationGlobalVBEM, n_neighbors_mi, distanceFunction_mi, gaussian_noise_mi, seed,
                    normalized_mi, initialVBEMConfig, localVBEMConfig, iterationVBEMConfig, finalVBEMConfig, typeLocalVBEM);
                break;
        }
    }

    public Tuple2<BayesianNetwork, Double> learnModel(DataOnMemory<DataInstance> data,
                                                      int alpha,
                                                      Map<String, double[]> priors,
                                                      LogUtils.LogLevel logLevel) {

        Result result = this.incLearner.learnModel(data, alpha, priors, logLevel);
        result.getPlateuStructure().updateParameterVariablesPrior(result.getPlateuStructure().getParameterVariablesPosterior());
        BayesianNetwork posteriorPredictive = new BayesianNetwork(result.getDag(), result.getPlateuStructure().getEFLearningBN().toConditionalDistribution());
        double elbo = result.getElbo();

        return new Tuple2<>(posteriorPredictive, elbo);
    }

    private void initializeHigh(boolean iterationGlobalVBEM,
                                int n_neighbors_mi,
                                DistanceFunction distanceFunction_mi,
                                boolean gaussian_noise_mi,
                                long seed,
                                boolean normalized_mi,
                                VBEMConfig initialVBEMConfig,
                                VBEMConfig localVBEMConfig,
                                VBEMConfig iterationVBEMConfig,
                                VBEMConfig finalVBEMConfig,
                                TypeLocalVBEM typeLocalVBEM) {

        boolean allowObservedToObserved = true;
        boolean allowObservedToLatent = true;

        Set<BlfmIncOperator> operators = new LinkedHashSet<>();
        BlfmIncAddDiscreteNode addDiscreteNodeOperator = new BlfmIncAddDiscreteNode(2, Integer.MAX_VALUE,
                localVBEMConfig, typeLocalVBEM);
        BlfmIncAddArc addArcOperator = new BlfmIncAddArc(allowObservedToObserved, allowObservedToLatent, allowObservedToObserved,
                localVBEMConfig, typeLocalVBEM);
        operators.add(addDiscreteNodeOperator);
        operators.add(addArcOperator);

        this.incLearner = new BLFM_IncLearner_Missing(operators, iterationGlobalVBEM, n_neighbors_mi, distanceFunction_mi,
                gaussian_noise_mi, seed, normalized_mi, initialVBEMConfig, localVBEMConfig, iterationVBEMConfig,
                finalVBEMConfig, typeLocalVBEM);
    }

    private void initializeMedium(boolean iterationGlobalVBEM,
                                  int n_neighbors_mi,
                                  DistanceFunction distanceFunction_mi,
                                  boolean gaussian_noise_mi,
                                  long seed,
                                  boolean normalized_mi,
                                  VBEMConfig initialVBEMConfig,
                                  VBEMConfig localVBEMConfig,
                                  VBEMConfig iterationVBEMConfig,
                                  VBEMConfig finalVBEMConfig,
                                  TypeLocalVBEM typeLocalVBEM) {

        boolean allowObservedToObserved = true;
        boolean allowObservedToLatent = false;

        Set<BlfmIncOperator> operators = new LinkedHashSet<>();
        BlfmIncAddDiscreteNode addDiscreteNodeOperator = new BlfmIncAddDiscreteNode(2, Integer.MAX_VALUE,
                localVBEMConfig, typeLocalVBEM);
        BlfmIncAddArc addArcOperator = new BlfmIncAddArc(allowObservedToObserved, allowObservedToLatent, allowObservedToObserved,
                localVBEMConfig, typeLocalVBEM);
        operators.add(addDiscreteNodeOperator);
        operators.add(addArcOperator);

        this.incLearner = new BLFM_IncLearner_Missing(operators, iterationGlobalVBEM, n_neighbors_mi, distanceFunction_mi,
                gaussian_noise_mi, seed, normalized_mi, initialVBEMConfig, localVBEMConfig, iterationVBEMConfig,
                finalVBEMConfig, typeLocalVBEM);
    }

    private void initializeLow(boolean iterationGlobalVBEM,
                               int n_neighbors_mi,
                               DistanceFunction distanceFunction_mi,
                               boolean gaussian_noise_mi,
                               long seed,
                               boolean normalized_mi,
                               VBEMConfig initialVBEMConfig,
                               VBEMConfig localVBEMConfig,
                               VBEMConfig iterationVBEMConfig,
                               VBEMConfig finalVBEMConfig,
                               TypeLocalVBEM typeLocalVBEM) {

        boolean allowObservedToObserved = false;
        boolean allowObservedToLatent = false;

        Set<BlfmIncOperator> operators = new LinkedHashSet<>();
        BlfmIncAddDiscreteNode addDiscreteNodeOperator = new BlfmIncAddDiscreteNode(2, Integer.MAX_VALUE,
                localVBEMConfig, typeLocalVBEM);
        BlfmIncAddArc addArcOperator = new BlfmIncAddArc(allowObservedToObserved, allowObservedToLatent, allowObservedToObserved,
                localVBEMConfig, typeLocalVBEM);
        operators.add(addDiscreteNodeOperator);
        operators.add(addArcOperator);

        this.incLearner = new BLFM_IncLearner_Missing(operators, iterationGlobalVBEM, n_neighbors_mi, distanceFunction_mi,
                gaussian_noise_mi, seed, normalized_mi, initialVBEMConfig, localVBEMConfig, iterationVBEMConfig,
                finalVBEMConfig, typeLocalVBEM);
    }
}
