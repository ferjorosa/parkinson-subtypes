package eu.amidst.extension.learn.structure;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.data.DataUtils;
import eu.amidst.extension.learn.parameter.VBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.VBEM_Global;
import eu.amidst.extension.learn.structure.operator.hc.tree.BltmHcDecreaseCard;
import eu.amidst.extension.learn.structure.operator.hc.tree.BltmHcIncreaseCard;
import eu.amidst.extension.learn.structure.operator.incremental.BlfmIncOperator;
import eu.amidst.extension.learn.structure.typelocalvbem.TypeLocalVBEM;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.distance.DistanceFunction;
import eu.amidst.extension.util.mi.MutualInformation;
import eu.amidst.extension.util.tuple.Tuple3;

import java.util.*;

// TODO: Antes de ejecutarlo es necesario probar que es capaz de seleccionar los alpha mejores pares de variables de la matriz de MIs
public class BLFM_IncLearner {

    private Set<BlfmIncOperator> operators;

    private boolean iterationGlobalVBEM;

    private int n_neighbors_mi;

    private DistanceFunction distanceFunction_mi;

    private boolean gaussianNoise_mi;

    private long gaussianNoiseSeed;

    private boolean normalizedMI;

    private VBEMConfig initialVBEMConfig;

    private VBEMConfig localVBEMConfig;

    private VBEMConfig iterationVBEMConfig;

    private VBEMConfig finalVBEMConfig;

    private TypeLocalVBEM typeLocalVBEM;

    public BLFM_IncLearner(Set<BlfmIncOperator> operators,
                           boolean iterationGlobalVBEM,
                           int n_neighbors_mi,
                           DistanceFunction distanceFunction_mi,
                           boolean gaussianNoise_mi,
                           long gaussianNoiseSeed,
                           boolean normalizedMI,
                           VBEMConfig initialVBEMConfig,
                           VBEMConfig localVBEMConfig,
                           VBEMConfig iterationVBEMConfig,
                           VBEMConfig finalVBEMConfig,
                           TypeLocalVBEM typeLocalVBEM) {
        this.operators = operators;
        this.iterationGlobalVBEM = iterationGlobalVBEM;
        this.n_neighbors_mi = n_neighbors_mi;
        this.distanceFunction_mi = distanceFunction_mi;
        this.gaussianNoise_mi = gaussianNoise_mi;
        this.gaussianNoiseSeed = gaussianNoiseSeed;
        this.normalizedMI = normalizedMI;
        this.initialVBEMConfig = initialVBEMConfig;
        this.localVBEMConfig = localVBEMConfig;
        this.iterationVBEMConfig = iterationVBEMConfig;
        this.finalVBEMConfig = finalVBEMConfig;
        this.typeLocalVBEM = typeLocalVBEM;
    }

    public Result learnModel(DataOnMemory<DataInstance> data, int alpha, Map<String, double[]> priors, LogUtils.LogLevel logLevel) {

        /* Inicializamos las estructuras necesarias */
        Variables variables = new Variables(data.getAttributes());
        DAG dag = new DAG(variables);

        Set<Variable> currentSet = new LinkedHashSet<>(); // Current set of variables being considered
        for(Variable variable: variables)
            currentSet.add(variable);

        /* Aprendemos el modelo inicial donde todas las variables son independientes y no hay latentes */
        VBEM initialVBEM = new VBEM(this.initialVBEMConfig);
        initialVBEM.learnModel(data, dag, priors);
        PlateuStructure currentModel = initialVBEM.getPlateuStructure();

        Result bestResult = new Result(currentModel, currentModel.getLogProbabilityOfEvidence(), dag, "BLFM_IncLearnerMax (alpha = " + alpha+")");

        /*
         * 1 - Estimamos la MI entre cada par de atributos de los datos
         *
         * Transformamos la matriz double[][] en un Map porque no sabemos cuantas variables latentes se van
         * a añadir y por tanto no sabemos el tamaño final de la matriz
         */
        double[][] mis = MutualInformation.estimate(data, this.n_neighbors_mi, this.distanceFunction_mi, this.gaussianNoise_mi, this.gaussianNoiseSeed, this.normalizedMI);

        // TODO: Estudiar como almacenarlo todo en la triangular superior
        /* "Matriz" de MIs. Segun se vayan eliminando variables del Set, lo haremos tambien de esta matriz */
        Map<Variable, Map<Variable, Double>> currentMIsMatrix = new LinkedHashMap<>(mis.length);
        for(int i=0; i < mis.length; i++) {
            Variable x = variables.getListOfVariables().get(i);
            currentMIsMatrix.put(x, new LinkedHashMap<>());
            for(int j = 0; j < mis.length; j++) {
                Variable y = variables.getListOfVariables().get(j);
                currentMIsMatrix.get(x).put(y, mis[i][j]);
            }
        }

        /* Inicializamos la estructura auxiliar para la estimacion de la MI de variables latentes */
        Map<Variable, double[]> currentDataForMI = new HashMap<>(variables.getNumberOfVars()); // Associated data to current set of variables. It will be used to estimate MI
        initializeDataForMI(currentDataForMI, data, variables);

        LogUtils.info("Initial score: " + bestResult.getElbo(), logLevel);

        /* 2 - Bucle principal */
        boolean keepsImproving = true;
        int iteration = 0;
        while(keepsImproving && currentSet.size() > 1) {

            iteration++;

            /* Estimamos los alpha pares de variables cuyo valor de MI es mas alto */
            PriorityQueue<Tuple3<Variable, Variable, Double>> selectedTriples = highestMiVariables(currentMIsMatrix, alpha);

            Result bestIterationResult = new Result(null, -Double.MAX_VALUE, null, "NONE");
            Tuple3<Variable, Variable, Result> bestIterationTriple = new Tuple3<>(null, null, bestIterationResult);

            /* 1.1 - Iterate through the operators and select the one that returns the best model */
            for (BlfmIncOperator operator : this.operators) {
                Tuple3<Variable, Variable, Result> operatorTriple = operator.apply(selectedTriples,
                        bestResult.getPlateuStructure(),
                        bestResult.getDag());

                double operatorScore = operatorTriple.getThird().getElbo();

                if(operatorTriple.getThird().getElbo() == -Double.MAX_VALUE)
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
            Variable miVar = null;

            if(!firstVar.isObservable() && firstVar.isDiscrete())
                latentVariables.add(firstVar.getName());

            if(!secondVar.isObservable() && secondVar.isDiscrete())
                latentVariables.add(secondVar.getName());

            /* 1.3 - Estimate cardinality and modify current set of variables */
            if (bestIterationTriple.getThird().getName().equals("AddDiscreteNode")) {
                Variable newLatentVar = bestIterationTriple.getThird().getDag().getParentSet(bestIterationTriple.getFirst()).getParents().get(0);
                latentVariables.add(newLatentVar.getName());
                /* Estimamos la cardinalidad de las variables latentes implicadas */
                bestIterationResult = estimateLocalCardinality(latentVariables, bestIterationResult.getDag(), bestIterationResult.getPlateuStructure());
                /* Eliminamos las variables hijas del currentSet y añadimos la latente nueva */
                removeVarFromCurrentDataStructures(firstVar, currentSet, currentDataForMI, currentMIsMatrix);
                removeVarFromCurrentDataStructures(secondVar, currentSet, currentDataForMI, currentMIsMatrix);
                currentSet.add(newLatentVar);
                /* Preparamos la variable padre para su nueva estimacion de MI */
                miVar = newLatentVar;

            } else if (bestIterationTriple.getThird().getName().equals("AddArc")) {
                /* Estimamos la cardinalidad de las variables latentes implicadas */
                bestIterationResult = estimateLocalCardinality(latentVariables, bestIterationResult.getDag(), bestIterationResult.getPlateuStructure());
                /* Eliminamos la variable hija del currentSet */
                removeVarFromCurrentDataStructures(secondVar, currentSet, currentDataForMI, currentMIsMatrix);
                /* Preparamos la variable padre para su nueva estimacion de MI */
                if(!firstVar.isObservable())
                    miVar = firstVar;
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
            /* En caso positivo, almacenamos el modelo de la iteracion y estimamos la MI de la variable padre */
            } else {
                LogUtils.debug("Improves the score: " + bestIterationResult.getElbo() + " > " + bestResult.getElbo() + " (old best)", logLevel);
                LogUtils.debug("--------------------------------------------------", logLevel);
                bestResult = bestIterationResult;
                if(miVar != null)
                    estimateVariableMIs(currentMIsMatrix, data, currentDataForMI, bestResult.getPlateuStructure(), bestResult.getDag(), miVar);
            }
        }

        /* 4 - Aprendemos el modelo de forma global y devolvemos la solucion */
        VBEM_Global finalVBEM = new VBEM_Global(this.finalVBEMConfig);
        finalVBEM.learnModel(bestResult.getPlateuStructure(), bestResult.getDag());

        PlateuStructure bestModel = finalVBEM.getPlateuStructure();
        double bestModelScore = bestModel.getLogProbabilityOfEvidence();
        DAG bestDAG = bestResult.getDag();

        LogUtils.info("\nFinal score after global VBEM: " + bestModelScore, logLevel);

        return new Result(bestModel, bestModelScore, bestDAG, "BLFM_IncLearnerMax (alpha = " + alpha+")");
    }

    private void initializeDataForMI(Map<Variable, double[]> currentData, DataOnMemory<DataInstance> data, Variables variables) {
        /* Map initialization */
        for(Variable variable: variables)
            currentData.put(variable, new double[data.getNumberOfDataInstances()]);

        /* Introduce values into the Map */
        for(int instIndex = 0; instIndex < data.getNumberOfDataInstances(); instIndex++) {
            double[] instData = data.getDataInstance(instIndex).toArray();
            for(int varIndex = 0; varIndex < variables.getNumberOfVars(); varIndex++) {
                Variable var = variables.getListOfVariables().get(varIndex);
                double[] currentVarData = currentData.get(var);
                currentVarData[instIndex] = instData[varIndex];
            }
        }
    }

    private PriorityQueue<Tuple3<Variable, Variable, Double>> highestMiVariables(Map<Variable, Map<Variable, Double>> misMatrix, int alpha) {

        PriorityQueue<Tuple3<Variable, Variable, Double>> queue = new PriorityQueue<>(alpha, new InverseMiComparator());

        /*
         * Creamos una lista con las keys del map para poder iterar por la "triangular" de la matriz.
         * Ademas, añadimos el primer elemento de la matriz a la queue para evitar tener que comprobar si la queue se encuentra vacia
         */
        List<Variable> keysList = new ArrayList<>(misMatrix.keySet());
        Variable firstKey = keysList.get(0);
        Variable secondKey = keysList.get(1);
        queue.add(new Tuple3<>(firstKey, secondKey, misMatrix.get(firstKey).get(secondKey)));

        /* Iteramos por la triangular de la matriz de MIs para obtener el par de variables con valor maximo */
        for(int i = 0; i < keysList.size(); i++)
            for(int j = i+1; j < keysList.size(); j++){
                Variable x = keysList.get(i);
                Variable y = keysList.get(j);

                if(queue.size() < alpha)
                    queue.add(new Tuple3<>(x,y,misMatrix.get(x).get(y)));
                else if(misMatrix.get(x).get(y) > queue.peek().getThird()){
                    queue.poll();
                    queue.add(new Tuple3<>(x,y,misMatrix.get(x).get(y)));
                }
            }

        return queue;
    }

    private class InverseMiComparator implements Comparator<Tuple3<Variable, Variable, Double>> {
        @Override
        public int compare(Tuple3<Variable, Variable, Double> o1, Tuple3<Variable, Variable, Double> o2) {
            if(o1.getThird().equals(o2.getThird()))
                return 0;
            // keep the biggest values
            return o1.getThird() > o2.getThird() ? 1 : -1;
        }
    }

    /** Estima la MI de la variable con las del resto del set. Esto sirve tanto para una nueva variable latente como para
     * actualizar los valores de una variable ya existente */
    private void estimateVariableMIs(Map<Variable, Map<Variable, Double>> misMatrix,
                                     DataOnMemory<DataInstance> data,
                                     Map<Variable, double[]> dataForMI,
                                     PlateuStructure currentModel,
                                     DAG currentDAG,
                                     Variable latentVar) {

        /* Obtenemos la posterior predictive. Para no interferir con el aprendizaje, realizamos una copia del Plateau */
        PlateuStructure modelCopy = currentModel.deepCopy(currentDAG);
        modelCopy.updateParameterVariablesPrior(modelCopy.getParameterVariablesPosterior());
        BayesianNetwork posteriorPredictive = new BayesianNetwork(currentDAG, modelCopy.getEFLearningBN().toConditionalDistribution());

        /* Completamos los datos de la nueva variable latente y los almacenamos en dataForMI */
        DataOnMemory<DataInstance> latentVarData = DataUtils.completeDiscreteLatent(data, posteriorPredictive, latentVar);
        double[] latentVarDataArray = new double[latentVarData.getNumberOfDataInstances()];
        dataForMI.put(latentVar, latentVarDataArray);

        /* Introduce values into the Map */
        for(int instIndex = 0; instIndex < latentVarData.getNumberOfDataInstances(); instIndex++) {
            double[] instData = latentVarData.getDataInstance(instIndex).toArray();
            latentVarDataArray[instIndex] = instData[0]; // Solo hay una variable latente
        }

        /* Estimamos la MI con todas las variables de misMatrix */
        misMatrix.put(latentVar, new LinkedHashMap<>());
        for(Variable var: misMatrix.keySet()){
            double[][] bothVarsData = generate2dMatrix(dataForMI.get(var), latentVarDataArray);
            double mi = MutualInformation.estimate(bothVarsData, var, latentVar, this.n_neighbors_mi, this.distanceFunction_mi, this.gaussianNoise_mi, this.gaussianNoiseSeed, this.normalizedMI);
            misMatrix.get(var).put(latentVar, mi);
            misMatrix.get(latentVar).put(var, mi);
        }
    }

    private double[][] generate2dMatrix(double[] firstVector, double[] secondVector) {
        double[][] matrix = new double[firstVector.length][2];

        for(int i = 0; i < firstVector.length; i++) {
            matrix[i][0] = firstVector[i];
            matrix[i][1] = secondVector[i];
        }

        return matrix;
    }

    /** Elimina una variable de: currentSet, currentDataForMI y currentMIsMatrix */
    private void removeVarFromCurrentDataStructures(Variable variable,
                                                    Set<Variable> currentSet,
                                                    Map<Variable, double[]> currentDataForMI,
                                                    Map<Variable, Map<Variable, Double>> currentMIsMatrix) {
        currentSet.remove(variable);
        currentDataForMI.remove(variable);
        currentMIsMatrix.remove(variable);
        for(Variable var: currentMIsMatrix.keySet())
            currentMIsMatrix.get(var).remove(variable);
    }

    /** Internal mini HC for estimating the cardinality of a list of latent variables using local VBEM */
    private Result estimateLocalCardinality(List<String> discreteLatentVars, DAG dag, PlateuStructure currentModel) {

        int maxCardinality = Integer.MAX_VALUE;
        BltmHcIncreaseCard increaseCardOperator = new BltmHcIncreaseCard(maxCardinality, this.localVBEMConfig, this.iterationVBEMConfig, typeLocalVBEM);
        BltmHcDecreaseCard decreaseCardOperator = new BltmHcDecreaseCard(2, this.localVBEMConfig, this.iterationVBEMConfig, typeLocalVBEM);

        Result bestResult = new Result(currentModel, currentModel.getLogProbabilityOfEvidence(), dag, "Initial");

        while (true) {
            Result increaseCardResult = increaseCardOperator.apply(bestResult.getPlateuStructure(), bestResult.getDag(), discreteLatentVars, false);
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
