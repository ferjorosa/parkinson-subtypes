package eu.amidst.extension.learn.structure;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.learn.parameter.VBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.VBEM_Global;
import eu.amidst.extension.learn.structure.operator.hc.tree.BltmHcDecreaseCard;
import eu.amidst.extension.learn.structure.operator.hc.tree.BltmHcIncreaseCard;
import eu.amidst.extension.learn.structure.operator.incremental.BlfmIncOperator;
import eu.amidst.extension.learn.structure.typelocalvbem.TypeLocalVBEM;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.tuple.Tuple3;

import java.util.*;

/** Brute force equivalent of the Incremental learner. The difference is that this method tries all pairs, not
 * a fixed number of them that have been ordered according to their respective MI.
 */
public class BLFM_IncLearnerMax {

    private Set<BlfmIncOperator> operators;

    private boolean iterationGlobalVBEM;

    private VBEMConfig initialVBEMConfig;

    private VBEMConfig localVBEMConfig;

    private VBEMConfig iterationVBEMConfig;

    private VBEMConfig finalVBEMConfig;

    private TypeLocalVBEM typeLocalVBEM;

    public BLFM_IncLearnerMax(Set<BlfmIncOperator> operators,
                              boolean iterationGlobalVBEM,
                              TypeLocalVBEM typeLocalVBEM) {
        this(operators,
                iterationGlobalVBEM,
                new VBEMConfig(),
                new VBEMConfig(),
                new VBEMConfig(),
                new VBEMConfig(),
                typeLocalVBEM);
    }

    public BLFM_IncLearnerMax(Set<BlfmIncOperator> operators,
                              boolean iterationGlobalVBEM,
                              VBEMConfig initialVBEMConfig,
                              VBEMConfig localVBEMConfig,
                              VBEMConfig iterationVBEMConfig,
                              VBEMConfig finalVBEMConfig,
                              TypeLocalVBEM typeLocalVBEM) {
        this.operators = operators;
        this.iterationGlobalVBEM = iterationGlobalVBEM;
        this.initialVBEMConfig = initialVBEMConfig;
        this.localVBEMConfig = localVBEMConfig;
        this.iterationVBEMConfig = iterationVBEMConfig;
        this.finalVBEMConfig = finalVBEMConfig;
        this.typeLocalVBEM = typeLocalVBEM;
    }

    public Result learnModel(DataOnMemory<DataInstance> data,
                             Map<String, double[]> priors,
                             LogUtils.LogLevel logLevel) {

        /* Inicializamos las estructuras necesarias */
        Variables variables = new Variables(data.getAttributes());
        DAG dag = new DAG(variables);

        Set<Variable> currentSet = new LinkedHashSet<>(); // Current set of variables being considered
        for(Variable variable: variables)
            currentSet.add(variable);

        /* Aprendemos el modelo inicial donde todas las variables son independientes y no hay latentes */
        VBEM vbem = new VBEM(this.initialVBEMConfig);
        double initialScore = vbem.learnModel(data, dag, priors);
        Result bestResult = new Result(vbem.getPlateuStructure(), initialScore, dag, "BLFM_IncLearnerMax");

        LogUtils.info("Initial score: " + bestResult.getElbo(), logLevel);

        /* 1 - Bucle principal */
        boolean keepsImproving = true;
        int iteration = 0;
        while(keepsImproving && currentSet.size() > 1) {

            iteration++;

            Result bestIterationResult = new Result(null, -Double.MAX_VALUE, null, "NONE");
            Tuple3<Variable, Variable, Result> bestIterationTriple = new Tuple3<>(null, null, bestIterationResult);

            /* 1.1 - Iterate through the operators and select the one that returns the best model */
            for (BlfmIncOperator operator : this.operators) {
                Tuple3<Variable, Variable, Result> operatorTriple = operator.apply(currentSet,
                        bestResult.getPlateuStructure(),
                        bestResult.getDag());

                double operatorScore = operatorTriple.getThird().getElbo();

                if(operatorScore == -Double.MAX_VALUE)
                    LogUtils.debug(operatorTriple.getThird().getName() + " -> NONE", logLevel);
                else
                    LogUtils.debug(operatorTriple.getThird().getName() + "(" + operatorTriple.getFirst().getName()+"," + operatorTriple.getSecond()+") -> " + operatorTriple.getThird().getElbo(), logLevel);

                if(operatorScore > bestIterationTriple.getThird().getElbo()) {
                    bestIterationTriple = operatorTriple;
                    bestIterationResult = bestIterationTriple.getThird();
                }
            }

            /* 1.2 - Select latent variables in the pair */
            List<String> latentVariables = new ArrayList<>();
            Variable firstVar = bestIterationTriple.getFirst();
            Variable secondVar = bestIterationTriple.getSecond();

            if(!firstVar.isObservable() && firstVar.isDiscrete())
                latentVariables.add(firstVar.getName());

            if(!secondVar.isObservable() && secondVar.isDiscrete())
                latentVariables.add(secondVar.getName());

            /* 1.3 - Estimate cardinality and modify current set of variables */
            if (bestIterationTriple.getThird().getName().equals("AddDiscreteNode")) {
                Variable newLatentVar = bestIterationTriple.getThird().getDag().getParentSet(bestIterationTriple.getFirst()).getParents().get(0);
                latentVariables.add(newLatentVar.getName());

                bestIterationResult = estimateLocalCardinality(latentVariables, bestIterationResult.getDag(), bestIterationResult.getPlateuStructure());

                currentSet.remove(bestIterationTriple.getFirst());
                currentSet.remove(bestIterationTriple.getSecond());
                currentSet.add(newLatentVar);

            } else if (bestIterationTriple.getThird().getName().equals("AddArc")) {
                bestIterationResult = estimateLocalCardinality(latentVariables, bestIterationResult.getDag(), bestIterationResult.getPlateuStructure());

                currentSet.remove(bestIterationTriple.getSecond());
            }

            /* 1.3 - Then, if allowed, we globally learn the parameters of the resulting model */
            if(this.iterationGlobalVBEM) {
                VBEM_Global iterationVBEM = new VBEM_Global(this.iterationVBEMConfig);
                iterationVBEM.learnModel(bestIterationResult.getPlateuStructure(), bestIterationResult.getDag());

                bestIterationResult = new Result(iterationVBEM.getPlateuStructure(),
                        iterationVBEM.getPlateuStructure().getLogProbabilityOfEvidence(),
                        bestIterationResult.getDag(),
                        bestIterationResult.getName());

                //LogUtils.printf("\nIteration score after global VBEM: " + bestIterationResult.getElbo(), debug);
            }

            LogUtils.info("\nIteration["+iteration+"] = "+bestIterationTriple.getThird().getName() +
                    "(" + bestIterationTriple.getFirst() + ", " + bestIterationTriple.getSecond() + ") -> " + bestIterationResult.getElbo(), logLevel);

            /* En caso de que la iteracion no consiga mejorar el score del modelo, paramos el bucle */
            if(bestIterationResult.getElbo() <= bestResult.getElbo()) {
                LogUtils.debug("Doesn't improve the score: " + bestIterationResult.getElbo() + " <= " + bestResult.getElbo() + " (old best)", logLevel);
                LogUtils.debug("--------------------------------------------------", logLevel);
                keepsImproving = false;
            } else {
                LogUtils.debug("Improves the score: " + bestIterationResult.getElbo() + " > " + bestResult.getElbo() + " (old best)", logLevel);
                LogUtils.debug("--------------------------------------------------", logLevel);
                bestResult = bestIterationResult;
            }
        }

        /* 4 - Aprendemos el modelo de forma global y devolvemos la solucion */
        VBEM_Global finalVBEM = new VBEM_Global(this.finalVBEMConfig);
        finalVBEM.learnModel(bestResult.getPlateuStructure(), bestResult.getDag());

        PlateuStructure bestModel = finalVBEM.getPlateuStructure();
        double bestModelScore = bestModel.getLogProbabilityOfEvidence();
        DAG bestDAG = bestResult.getDag();

        LogUtils.info("\nFinal score after global VBEM: " + bestModelScore, logLevel);

        return new Result(bestModel, bestModelScore, bestDAG, "BLFM_IncLearnerMax");
    }

    /** Internal mini HC for estimating the cardinality of a list of latent variables using local VBEM */
    private Result estimateLocalCardinality(List<String> discreteLatentVars, DAG dag, PlateuStructure currentModel) {

        int maxCardinality = Integer.MAX_VALUE;
        BltmHcIncreaseCard increaseCardOperator = new BltmHcIncreaseCard(maxCardinality, this.localVBEMConfig, this.iterationVBEMConfig, typeLocalVBEM);
        BltmHcDecreaseCard decreaseCardOperator = new BltmHcDecreaseCard(2, this.localVBEMConfig, this.iterationVBEMConfig, typeLocalVBEM);

        Result bestResult = new Result(currentModel, currentModel.getLogProbabilityOfEvidence(), dag, "Initial");

        while (true) {
            Result increaseCardResult = increaseCardOperator.apply(bestResult.getPlateuStructure(), bestResult.getDag(), discreteLatentVars,  false);
            Result decreaseCardResult = decreaseCardOperator.apply(bestResult.getPlateuStructure(), bestResult.getDag(), discreteLatentVars, false);

            if(increaseCardResult.getElbo() > decreaseCardResult.getElbo() && increaseCardResult.getElbo() > bestResult.getElbo())
                bestResult = increaseCardResult;
            else if(decreaseCardResult.getElbo() > increaseCardResult.getElbo() && decreaseCardResult.getElbo() > bestResult.getElbo())
                bestResult = decreaseCardResult;
            else
                return bestResult;
        }
    }
}
