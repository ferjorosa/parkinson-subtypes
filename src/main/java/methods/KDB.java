package methods;

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
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.tuple.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

// TODO: Introduce arc restrictions in VBSEM
// TODO: El generador de matrices de ayacencia tiene que tener en cuenta las siguientes restricciones:
//          - Se deben mantener los arcos de las restricciones
//          - No se puede superar el numero máximo de padres establecido por el usuario
// TODO: Por el momento el VBSEM solo trabaja con Empirical Bayes, asi que las priors estan limitadas
// NOTA: El KDB crea en cada iteracion un LCM que utiliza como estructura de partida para la inicializacion
public class KDB {

    private VBEMConfig vbemConfig;

    private BayesianHcConfig bayesianHcConfig;

    private int maxIterationsVBSEM;

    private InitializationVBSEM initializationVBSEM;

    public KDB(VBEMConfig vbemConfig, BayesianHcConfig bayesianHcConfig, int maxIterationsVBSEM, InitializationVBSEM initializationVBSEM) {
        this.vbemConfig = vbemConfig;
        this.bayesianHcConfig = bayesianHcConfig;
        this.maxIterationsVBSEM = maxIterationsVBSEM;
        this.initializationVBSEM = initializationVBSEM;
    }

    public Tuple2<BayesianNetwork, Double> learnModel(DataOnMemory<DataInstance> data,
                                                      int cardinality,
                                                      Map<String, double[]> priors,
                                                      LogUtils.LogLevel vbsemLogLevel,
                                                      LogUtils.LogLevel hcLogLevel) {

        /* Create an LCM that will be the base model and learn its parameters */
        DAG lcmDag = generateLCM(data, "LatentVar", cardinality);
        Variable latentVar = lcmDag.getVariables().getVariableByName("LatentVar");
        VBEM vbem = new VBEM(this.vbemConfig);
        double lcmScore = vbem.learnModelWithPriorUpdate(data, lcmDag, priors);
        BayesianNetwork lcm = vbem.getLearntBayesianNetwork();

        /* Create the Bayesian HC operators */
        Map<Variable, List<Variable>> cannotRemoveArcs = new HashMap<>();
        List<Variable> observableVars = lcmDag.getVariables().getListOfVariables().stream().filter(Variable::isObservable).collect(Collectors.toList());
        cannotRemoveArcs.put(latentVar, observableVars);

        BayesianHcAddArc bayesianHcAddArc = new BayesianHcAddArc(bayesianHcConfig, lcmDag.getVariables(), 3);
        BayesianHcRemoveArc bayesianHcRemoveArc = new BayesianHcRemoveArc(bayesianHcConfig, cannotRemoveArcs);
        BayesianHcReverseArc bayesianHcReverseArc = new BayesianHcReverseArc(bayesianHcConfig, lcmDag.getVariables(), 3);
        Set<BayesianHcOperator> bayesianHcOperators = new LinkedHashSet<>(3);
        bayesianHcOperators.add(bayesianHcAddArc);
        bayesianHcOperators.add(bayesianHcRemoveArc);
        bayesianHcOperators.add(bayesianHcReverseArc);

        /* Create the VBSEM with restrictions */
        VBSEM_restrictions vbsem = new VBSEM_restrictions(vbemConfig, bayesianHcConfig, initializationVBSEM, maxIterationsVBSEM);

        /* With current cardinality, apply VBSEM to learn the best structure for observable variables (bridge is fixed) */
        return vbsem.learnModel(lcm, lcmScore, data, bayesianHcOperators, vbsemLogLevel, hcLogLevel);
    }

    public Tuple2<BayesianNetwork, Double> learnModelIteratively(DataOnMemory<DataInstance> data,
                                                                 int startCardinality,
                                                                 int maxCardinality,
                                                                 Map<String, double[]> priors,
                                                                 LogUtils.LogLevel logLevel,
                                                                 LogUtils.LogLevel vbsemLogLevel,
                                                                 LogUtils.LogLevel hcLogLevel) {

        BayesianNetwork bestModel = null;
        double bestScore = -Double.MAX_VALUE;

        for(int card = startCardinality; card <= maxCardinality; card++) {
            Tuple2<BayesianNetwork, Double> result = learnModel(data, card, priors, vbsemLogLevel, hcLogLevel);

            if(result.getSecond() > bestScore) {
                bestModel = result.getFirst();
                bestScore = result.getSecond();

                LogUtils.info("\n(" + card + "): " + result.getSecond() + " -> SCORE IMPROVEMENT\n", logLevel);
            } else {
                LogUtils.info("\n(" + card + "): " + result.getSecond() + " -> STOP\n", logLevel);
                return new Tuple2<>(bestModel, bestScore);
            }
        }

        return new Tuple2<>(bestModel, bestScore);
    }

    private DAG generateLCM(DataOnMemory<DataInstance> data, String latentVarName, int cardinality) {

        /* Create a Naïve Bayes structure with a latent parent */
        Variables variables = new Variables(data.getAttributes());
        Variable latentVar = variables.newMultinomialVariable(latentVarName, cardinality);

        DAG dag = new DAG(variables);

        for(Variable var: variables)
            if(!var.equals(latentVar))
                dag.getParentSet(var).addParent(latentVar);

        return dag;
    }
}
