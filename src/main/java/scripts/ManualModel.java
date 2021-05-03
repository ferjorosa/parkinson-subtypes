package scripts;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.data.DataUtils;
import eu.amidst.extension.learn.parameter.InitializationTypeVBEM;
import eu.amidst.extension.learn.parameter.InitializationVBEM;
import eu.amidst.extension.learn.parameter.VBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.penalizer.BishopPenalizer;
import eu.amidst.extension.util.EstimatePredictiveScore;
import eu.amidst.extension.util.PriorsFromData;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/** The objective of this script is to test if Tremor and rigidity have independent subtypes. For this reason, we
 * are first going to manually construct the original GLSL_CIL_1 network and then try different options to analyse
 * the existence of these subtypes in the data.
 *
 * */
public class ManualModel {

    public static void main(String[] args) throws Exception {


        DataOnMemory<DataInstance> data = DataStreamLoader.loadDataOnMemoryFromFile("data/data_numerical.arff");
        // Data completed by the original model GLSL_CIL_1
        DataOnMemory<DataInstance> completeData = DataStreamLoader.loadDataOnMemoryFromFile("results/glsl_cil_1_64.arff");

        /* Filter the socio-demographic attributes */
        List<Attribute> symptomAttributes = data.getAttributes().getFullListOfAttributes().stream()
                .filter(x->
                        !x.getName().equals("patnum") &&
                                !x.getName().equals("age") &&
                                !x.getName().equals("sex") &&
                                !x.getName().equals("pdonset") &&
                                !x.getName().equals("durat_pd") &&
                                !x.getName().equals("hy")
                )
                .collect(Collectors.toList());
        data = DataUtils.project(data, symptomAttributes);

        /* Generate Empirical Bayes priors from data, ignoring missing values */
        Map<String, double[]> priors = PriorsFromData.generate(completeData, 1);

        /* Generate DAG */
        DAG originalDAG = originalGraph(data);
        DAG originalWithTremorAndRigiditySubtypesDAG = originalWithTremorAndRigiditySubtypes(data);

        /* Learn model parameters with VBEM */
        //learnModel(data, originalDAG, priors, "GLSL-CIL");
        //learnModel(data, originalWithTremorSubtypeDAG, priors, "GLSL-CIL with Tremor");
        learnModel(data, originalWithTremorAndRigiditySubtypesDAG, priors, "GLSL-CIL with Tremor and Rigidity");
    }

    private static void learnModel(DataOnMemory<DataInstance> data, DAG dag, Map<String, double[]> priors, String name) {

        System.out.println("\n==================================");
        System.out.println("============ "+name+" ============");
        System.out.println("==================================");

        long seed = 0;
        int nVbemCandidates = 64;

        /* Learn model parameters with VBEM */
        InitializationVBEM initializationVBEM = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, nVbemCandidates, nVbemCandidates/2, true);
        VBEMConfig vbemConfig = new VBEMConfig(seed, 0.01, 100, initializationVBEM, new BishopPenalizer());
        VBEM vbem = new VBEM(vbemConfig);

        double elbo = vbem.learnModelWithPriorUpdate(data, dag, priors);
        BayesianNetwork posteriorPredictive = vbem.getLearntBayesianNetwork();
        double logLikelihood = EstimatePredictiveScore.amidstLL(posteriorPredictive, data);
        double bic = EstimatePredictiveScore.amidstBIC(posteriorPredictive, data);
        double aic = EstimatePredictiveScore.amidstAIC(posteriorPredictive, data);

        System.out.println("ELBO: " + elbo);
        System.out.println("LogLikelihood: " + logLikelihood);
        System.out.println("BIC: " + bic);
        System.out.println("AIC: " + aic);
    }

    /** Manually construct the graph from GLSL_CIL_1 */
    private static DAG originalGraph(DataOnMemory<DataInstance> data) {

        Variables variables = new Variables(data.getAttributes());
        Variable A = variables.newMultinomialVariable("A", 2);
        Variable B = variables.newMultinomialVariable("B", 2);
        Variable C = variables.newMultinomialVariable("C", 3);
        Variable D = variables.newMultinomialVariable("D", 2);
        Variable E = variables.newMultinomialVariable("E", 2);
        Variable F = variables.newMultinomialVariable("F", 2);
        Variable G = variables.newMultinomialVariable("G", 2);
        Variable H = variables.newMultinomialVariable("H", 2);
        Variable I = variables.newMultinomialVariable("I", 2);

        DAG originalDag = new DAG(variables);

        originalDag.getParentSet(variables.getVariableByName("impulse_control")).addParent(A);
        originalDag.getParentSet(variables.getVariableByName("pigd")).addParent(A);

        originalDag.getParentSet(variables.getVariableByName("apathy")).addParent(B);
        originalDag.getParentSet(variables.getVariableByName("cognition")).addParent(B);
        originalDag.getParentSet(variables.getVariableByName("urinary")).addParent(B);
        originalDag.getParentSet(variables.getVariableByName("gastrointestinal")).addParent(B);
        originalDag.getParentSet(variables.getVariableByName("sleep")).addParent(B);
        originalDag.getParentSet(variables.getVariableByName("pain")).addParent(B);
        originalDag.getParentSet(B).addParent(A);

        originalDag.getParentSet(variables.getVariableByName("dyskinesias")).addParent(C);
        originalDag.getParentSet(variables.getVariableByName("psychosis")).addParent(C);
        originalDag.getParentSet(C).addParent(B);

        originalDag.getParentSet(variables.getVariableByName("mental_fatigue")).addParent(D);
        originalDag.getParentSet(variables.getVariableByName("physical_tiredness")).addParent(variables.getVariableByName("mental_fatigue"));
        originalDag.getParentSet(D).addParent(B);

        originalDag.getParentSet(variables.getVariableByName("fluctuations")).addParent(E);
        originalDag.getParentSet(variables.getVariableByName("axial_no_pigd")).addParent(E);
        originalDag.getParentSet(variables.getVariableByName("smell")).addParent(variables.getVariableByName("fluctuations"));
        originalDag.getParentSet(variables.getVariableByName("bradykinesia")).addParent(variables.getVariableByName("axial_no_pigd"));
        originalDag.getParentSet(E).addParent(D);

        originalDag.getParentSet(variables.getVariableByName("hypotension")).addParent(F);
        originalDag.getParentSet(variables.getVariableByName("sexual")).addParent(F);
        originalDag.getParentSet(F).addParent(B);

        originalDag.getParentSet(variables.getVariableByName("depression")).addParent(G);
        originalDag.getParentSet(variables.getVariableByName("weight_loss")).addParent(G);
        originalDag.getParentSet(G).addParent(B);
        originalDag.getParentSet(G).addParent(I);

        originalDag.getParentSet(variables.getVariableByName("sweating")).addParent(H);
        originalDag.getParentSet(variables.getVariableByName("anxiety")).addParent(H);
        originalDag.getParentSet(H).addParent(I);

        return originalDag;
    }

    private static DAG originalWithTremorAndRigiditySubtypes(DataOnMemory<DataInstance> data) {

        Variables variables = new Variables(data.getAttributes());
        Variable A = variables.newMultinomialVariable("A", 2);
        Variable B = variables.newMultinomialVariable("B", 2);
        Variable C = variables.newMultinomialVariable("C", 3);
        Variable D = variables.newMultinomialVariable("D", 2);
        Variable E = variables.newMultinomialVariable("E", 2);
        Variable F = variables.newMultinomialVariable("F", 2);
        Variable G = variables.newMultinomialVariable("G", 2);
        Variable H = variables.newMultinomialVariable("H", 2);
        Variable I = variables.newMultinomialVariable("I", 2);
        Variable J = variables.newMultinomialVariable("J", 2);
        Variable K = variables.newMultinomialVariable("K", 2);

        DAG originalDag = new DAG(variables);

        originalDag.getParentSet(variables.getVariableByName("impulse_control")).addParent(A);
        originalDag.getParentSet(variables.getVariableByName("pigd")).addParent(A);

        originalDag.getParentSet(variables.getVariableByName("apathy")).addParent(B);
        originalDag.getParentSet(variables.getVariableByName("cognition")).addParent(B);
        originalDag.getParentSet(variables.getVariableByName("urinary")).addParent(B);
        originalDag.getParentSet(variables.getVariableByName("gastrointestinal")).addParent(B);
        originalDag.getParentSet(variables.getVariableByName("sleep")).addParent(B);
        originalDag.getParentSet(variables.getVariableByName("pain")).addParent(B);
        originalDag.getParentSet(B).addParent(A);

        originalDag.getParentSet(variables.getVariableByName("dyskinesias")).addParent(C);
        originalDag.getParentSet(variables.getVariableByName("psychosis")).addParent(C);
        originalDag.getParentSet(C).addParent(B);

        originalDag.getParentSet(variables.getVariableByName("mental_fatigue")).addParent(D);
        originalDag.getParentSet(variables.getVariableByName("physical_tiredness")).addParent(variables.getVariableByName("mental_fatigue"));
        originalDag.getParentSet(D).addParent(B);

        originalDag.getParentSet(variables.getVariableByName("fluctuations")).addParent(E);
        originalDag.getParentSet(variables.getVariableByName("axial_no_pigd")).addParent(E);
        originalDag.getParentSet(variables.getVariableByName("smell")).addParent(variables.getVariableByName("fluctuations"));
        originalDag.getParentSet(variables.getVariableByName("bradykinesia")).addParent(variables.getVariableByName("axial_no_pigd"));
        originalDag.getParentSet(E).addParent(D);

        originalDag.getParentSet(variables.getVariableByName("hypotension")).addParent(F);
        originalDag.getParentSet(variables.getVariableByName("sexual")).addParent(F);
        originalDag.getParentSet(F).addParent(B);

        originalDag.getParentSet(variables.getVariableByName("depression")).addParent(G);
        originalDag.getParentSet(variables.getVariableByName("weight_loss")).addParent(G);
        originalDag.getParentSet(G).addParent(B);
        originalDag.getParentSet(G).addParent(I);

        originalDag.getParentSet(variables.getVariableByName("sweating")).addParent(H);
        originalDag.getParentSet(variables.getVariableByName("anxiety")).addParent(H);
        originalDag.getParentSet(H).addParent(I);

        originalDag.getParentSet(variables.getVariableByName("tremor")).addParent(J);

        originalDag.getParentSet(variables.getVariableByName("rigidity")).addParent(K);

        return originalDag;
    }
}
