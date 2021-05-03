package methods;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.learn.parameter.VBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.tuple.Tuple2;

import java.util.Map;

public class LCM {

    private VBEMConfig vbemConfig;

    public LCM(VBEMConfig vbemConfig) {
        this.vbemConfig = vbemConfig;
    }

    public Tuple2<BayesianNetwork, Double> learnModel(DataOnMemory<DataInstance> data,
                                                      int cardinality,
                                                      Map<String, double[]> priors) {

        VBEM vbem = new VBEM(this.vbemConfig);
        DAG lcmDag = generateLCM(data, "latentVar", cardinality);
        double score = vbem.learnModelWithPriorUpdate(data, lcmDag, priors);
        BayesianNetwork model = vbem.getLearntBayesianNetwork();

        return new Tuple2<>(model, score);
    }

    public Tuple2<BayesianNetwork, Double> learnModelIteratively(DataOnMemory<DataInstance> data,
                                                                 int startCardinality,
                                                                 int maxCardinality,
                                                                 Map<String, double[]> priors,
                                                                 LogUtils.LogLevel logLevel) {

        BayesianNetwork bestModel = null;
        double bestScore = -Double.MAX_VALUE;

        for(int card = startCardinality; card <= maxCardinality; card++) {
            Tuple2<BayesianNetwork, Double> result = learnModel(data, card, priors);

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
