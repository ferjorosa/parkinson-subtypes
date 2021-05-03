package eu.amidst.extension.learn.structure.glsl.operator;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.learn.parameter.VBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHcConfig;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcAddArc;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcOperator;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcRemoveArc;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcReverseArc;
import eu.amidst.extension.learn.structure.vbsem.InitializationVBSEM;
import eu.amidst.extension.learn.structure.vbsem.VBSEM_restrictions;
import eu.amidst.extension.util.GraphUtilsAmidst;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.PriorsFromData;
import eu.amidst.extension.util.tuple.Tuple2;
import eu.amidst.extension.util.tuple.Tuple3;
import org.apache.commons.math3.util.Combinations;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Procedure:
 *
 * 1 - Select those variables in the model with no parent
 * 2 - Generate the set of possible combinations
 */
public class GLSL_AddLatent_child implements GLSL_Operator {

    private int maxNumberOfLatentNodes;

    private int maxNumberParents_latent;

    private int maxNumberParents_observed;

    private VBEMConfig vbemConfig;

    private BayesianHcConfig bayesianHcConfig;

    private InitializationVBSEM initializationVBSEM;

    private int latentVarNameCounter = 0;

    public GLSL_AddLatent_child(int maxNumberOfDiscreteLatentNodes,
                                int maxNumberParents_latent,
                                int maxNumberParents_observed,
                                VBEMConfig vbemConfig,
                                BayesianHcConfig bayesianHcConfig,
                                InitializationVBSEM initializationVBSEM) {
        this.maxNumberOfLatentNodes = maxNumberOfDiscreteLatentNodes;
        this.maxNumberParents_latent = maxNumberParents_latent;
        this.maxNumberParents_observed = maxNumberParents_observed;
        this.vbemConfig = vbemConfig;
        this.bayesianHcConfig = bayesianHcConfig;
        this.initializationVBSEM = initializationVBSEM;
    }

    public GLSL_AddLatent_child(int maxNumberOfDiscreteLatentNodes,
                                int maxNumberParents_latent,
                                int maxNumberParents_observed,
                                VBEMConfig vbemConfig,
                                BayesianHcConfig bayesianHcConfig,
                                InitializationVBSEM initializationVBSEM,
                                int latentVarNameCounter) {
        this.maxNumberOfLatentNodes = maxNumberOfDiscreteLatentNodes;
        this.maxNumberParents_latent = maxNumberParents_latent;
        this.maxNumberParents_observed = maxNumberParents_observed;
        this.vbemConfig = vbemConfig;
        this.bayesianHcConfig = bayesianHcConfig;
        this.initializationVBSEM = initializationVBSEM;
        this.latentVarNameCounter = latentVarNameCounter;
    }

    @Override
    public Tuple3<String, BayesianNetwork, Double> apply(DAG dag, DataOnMemory<DataInstance> data, LogUtils.LogLevel logLevel) {

        BayesianNetwork bestModel = null;
        double bestModelScore = -Double.MAX_VALUE;

        /* Iterate through the set of latent variables */
        for(Variable variable: dag.getVariables()) {

            if (!variable.isObservable()) {

                List<Variable> children = GraphUtilsAmidst.getChildren(variable, dag);

                if (children.size() > 2) {

                    List<Tuple2<String, String>> childrenCombinations = generateVariableCombinations(children);

                    for(Tuple2<String, String> combination: childrenCombinations) {

                        Variables copyVariables = dag.getVariables().deepCopy();
                        DAG copyDag = dag.deepCopy(copyVariables);
                        Variable copyVariable = copyVariables.getVariableByName(variable.getName());

                        /*
                         * - Create a new latent variable with cardinality equal to the parent's cardinality
                         * - The selected pair of variables are his new children
                         * */
                        Variable copyChild_1 = copyVariables.getVariableByName(combination.getFirst());
                        Variable copyChild_2 = copyVariables.getVariableByName(combination.getSecond());

                        Variable newLatentVar = copyVariables.newMultinomialVariable("LV_" + (this.latentVarNameCounter++), copyVariable.getNumberOfStates());
                        copyDag.addVariable(newLatentVar);
                        copyDag.getParentSet(copyChild_1).removeParent(copyVariable);
                        copyDag.getParentSet(copyChild_2).removeParent(copyVariable);

                        copyDag.getParentSet(copyChild_1).addParent(newLatentVar);
                        copyDag.getParentSet(copyChild_2).addParent(newLatentVar);
                        copyDag.getParentSet(newLatentVar).addParent(copyVariable);

                        /* Apply VBEM to generate a base model for VBSEM */
                        Map<String, double[]> priors = PriorsFromData.generate(data, 1);
                        VBEM vbem = new VBEM(vbemConfig);
                        double baseScore = vbem.learnModelWithPriorUpdate(data, copyDag, priors);
                        BayesianNetwork baseModel = vbem.getLearntBayesianNetwork();

                        /* Apply VBSEM to generate a new model */
                        Set<BayesianHcOperator> bayesianHcOperators = initializeBayesianHcOperators(baseModel);
                        VBSEM_restrictions vbsem_restrictions = new VBSEM_restrictions(vbemConfig, bayesianHcConfig, initializationVBSEM, 100);
                        Tuple2<BayesianNetwork, Double> result = vbsem_restrictions.learnModel(baseModel, baseScore, data, bayesianHcOperators, LogUtils.LogLevel.NONE, LogUtils.LogLevel.NONE);

                        LogUtils.info("AL " + newLatentVar + ", child of " + copyVariable + " -> " + result.getSecond(), logLevel);

                        if(result.getSecond() > bestModelScore) {
                            bestModel = result.getFirst();
                            bestModelScore = result.getSecond();
                        }
                    }
                }
            }
        }

        return new Tuple3<>("AddLatent", bestModel, bestModelScore);
    }

    private List<Tuple2<String, String>> generateVariableCombinations(List<Variable> variables) {

        List<Tuple2<String, String>> variableCombinations = new ArrayList<>();
        Iterator<int[]> variableIndexCombinations = new Combinations(variables.size(), 2).iterator();

        /* Iteramos por las combinaciones no repetidas de variables hijas y generamos una Tuple con cada una */
        while(variableIndexCombinations.hasNext()) {
            // Indices de los clusters a comparar
            int[] combination = variableIndexCombinations.next();
            // Añadimos la nueva tupla
            variableCombinations.add(new Tuple2<>(variables.get(combination[0]).getName(), variables.get(combination[1]).getName()));
        }

        return variableCombinations;
    }

    /** Initialize Hc operators with the default arc restrictions discussed at the beginning of the file */
    private Set<BayesianHcOperator> initializeBayesianHcOperators(BayesianNetwork model) {
        List<Variable> latentVars = model.getVariables().getListOfVariables().stream().filter(x->!x.isObservable()).collect(Collectors.toList());
        List<Variable> observedVars = model.getVariables().getListOfVariables().stream().filter(x->x.isObservable()).collect(Collectors.toList());
        Map<Variable, List<Variable>> cannotAddOrReverseArcs = new HashMap<>();
        for(Variable observedVar: observedVars)
            cannotAddOrReverseArcs.put(observedVar, latentVars);

        Map<Variable, Integer> maxNumberParents = new HashMap<>();
        for(Variable latentVar: latentVars)
            maxNumberParents.put(latentVar, this.maxNumberParents_latent);
        for(Variable observedVar: observedVars)
            maxNumberParents.put(observedVar, this.maxNumberParents_observed);

        BayesianHcAddArc bayesianHcAddArc = new BayesianHcAddArc(bayesianHcConfig, model.getVariables(), cannotAddOrReverseArcs, maxNumberParents);
        BayesianHcRemoveArc bayesianHcRemoveArc = new BayesianHcRemoveArc(bayesianHcConfig, new HashMap<>());
        BayesianHcReverseArc bayesianHcReverseArc = new BayesianHcReverseArc(bayesianHcConfig, model.getVariables(), cannotAddOrReverseArcs, maxNumberParents);

        Set<BayesianHcOperator> bayesianHcOperators = new LinkedHashSet<>(3);
        bayesianHcOperators.add(bayesianHcAddArc);
        bayesianHcOperators.add(bayesianHcRemoveArc);
        bayesianHcOperators.add(bayesianHcReverseArc);

        return bayesianHcOperators;
    }
}
