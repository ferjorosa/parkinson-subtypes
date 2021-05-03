package eu.amidst.extension.learn.structure;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.core.variables.stateSpaceTypes.FiniteStateSpace;
import eu.amidst.extension.data.DataUtils;
import eu.amidst.extension.learn.parameter.VBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.VBEM_Global;
import eu.amidst.extension.learn.parameter.VBEM_Local;
import eu.amidst.extension.learn.structure.typelocalvbem.TypeLocalVBEM;
import eu.amidst.extension.util.GraphUtilsAmidst;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.distance.DistanceFunction;
import eu.amidst.extension.util.mi.MutualInformation;
import eu.amidst.extension.util.tuple.Tuple2;

import java.util.*;

// TODO: En principio no seria necesario tener dataForMI ya que solo se estima la MI una vez al ser arboles binarios
public class BLFM_BinG {

    private int n_neighbors_mi;

    private DistanceFunction distanceFunction_mi;

    private boolean gaussianNoise_mi;

    private long gaussianNoiseSeed;

    private boolean normalizedMI;

    private VBEMConfig initialVBEMConfig;

    private VBEMConfig localVBEMConfig;

    private VBEMConfig finalVBEMConfig;

    TypeLocalVBEM typeLocalVBEM;

    private int latentVarNameCounter = 0;

    public BLFM_BinG(int n_neighbors_mi,
                     DistanceFunction distanceFunction_mi,
                     boolean gaussianNoise_mi,
                     long gaussianNoiseSeed,
                     boolean normalizedMI,
                     TypeLocalVBEM typeLocalVBEM) {
        this(n_neighbors_mi,
                distanceFunction_mi,
                gaussianNoise_mi,
                gaussianNoiseSeed,
                normalizedMI,
                new VBEMConfig(),
                new VBEMConfig(),
                new VBEMConfig(),
                typeLocalVBEM);
    }

    public BLFM_BinG(int n_neighbors_mi,
                     DistanceFunction distanceFunction_mi,
                     boolean gaussianNoise_mi,
                     long gaussianNoiseSeed,
                     boolean normalizedMI,
                     VBEMConfig initialVBEMConfig,
                     VBEMConfig localVBEMConfig,
                     VBEMConfig finalVBEMConfig,
                     TypeLocalVBEM typeLocalVBEM) {
        this.n_neighbors_mi = n_neighbors_mi;
        this.distanceFunction_mi = distanceFunction_mi;
        this.gaussianNoise_mi = gaussianNoise_mi;
        this.gaussianNoiseSeed = gaussianNoiseSeed;
        this.normalizedMI = normalizedMI;
        this.initialVBEMConfig = initialVBEMConfig;
        this.localVBEMConfig = localVBEMConfig;
        this.finalVBEMConfig = finalVBEMConfig;
        this.typeLocalVBEM = typeLocalVBEM;
    }

    public Result learnModel(DataOnMemory<DataInstance> data, Map<String, double[]> priors, LogUtils.LogLevel logLevel) {

        /* Inicializamos las estructuras necesarias */
        Variables variables = new Variables(data.getAttributes());
        DAG dag = new DAG(variables);

        Set<Variable> currentSet = new LinkedHashSet<>(); // Current set of variables being considered
        for(Variable variable: variables)
            currentSet.add(variable);

        /* Aprendemos el modelo inicial donde todas las variables son independientes y no hay latentes */
        VBEM vbem = new VBEM(this.initialVBEMConfig);
        vbem.learnModel(data, dag, priors);
        PlateuStructure currentModel = vbem.getPlateuStructure();

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

        LogUtils.info("Initial score: " + currentModel.getLogProbabilityOfEvidence(), logLevel);

        /* 2 - Bucle principal */
        boolean keepsImproving = true;
        int iteration = 0;
        while(keepsImproving && currentSet.size() > 1) {

            iteration++;

            /* 3 - Escogemos el par de variables con mayor MI */
            Tuple2<Variable, Variable> highestMiVariablesIndexes = highestMiVariables(currentMIsMatrix);
            Variable firstVar = highestMiVariablesIndexes.getFirst();
            Variable secondVar = highestMiVariablesIndexes.getSecond();

            Variable newLatentVar = variables.newMultinomialVariable("LV_" + (this.latentVarNameCounter++), 2);
            dag.addVariable(newLatentVar);
            dag.getParentSet(firstVar).addParent(newLatentVar);
            dag.getParentSet(secondVar).addParent(newLatentVar);

            /* Creamos un nuevo Plateau para el aprendizaje donde omitimos la nueva variable latente y sus hijos */
            HashSet<Variable> omittedVariables = new HashSet<>();
            omittedVariables.add(newLatentVar);
            omittedVariables.addAll(GraphUtilsAmidst.getChildren(newLatentVar, dag));
            PlateuStructure copyPlateauStructure = currentModel.deepCopy(dag, omittedVariables);

            /* Aprendemos el modelo de forma local */
            VBEM_Local vbem_local = new VBEM_Local(this.localVBEMConfig);
            vbem_local.learnModel(copyPlateauStructure, dag, typeLocalVBEM.variablesToUpdate(newLatentVar, dag));

            LogUtils.info("\nIteration["+iteration+"] = AddDiscreteNode (" + firstVar + ", " + secondVar + ") -> " + vbem_local.getPlateuStructure().getLogProbabilityOfEvidence(), logLevel);

            /* Comparamos el score del nuevo modelo con el del actual. En caso positivo: */
            if (vbem_local.getPlateuStructure().getLogProbabilityOfEvidence() > currentModel.getLogProbabilityOfEvidence()) {
                /* 1 - La nueva variable latente pasa a formar parte del modelo actual */
                currentModel = vbem_local.getPlateuStructure();
                /* 2 - Estimamos la cardinalidad de la nueva variable latente */
                currentModel = greedyCardIncrease(newLatentVar, currentModel, dag);
                /* 3 - Eliminamos las variables observadas del currentSet y añadimos la latente nueva */
                removeVarFromCurrentDataStructures(firstVar, currentSet, currentDataForMI, currentMIsMatrix);
                removeVarFromCurrentDataStructures(secondVar, currentSet, currentDataForMI, currentMIsMatrix);
                currentSet.add(newLatentVar);
                /* 4 - Estimamos la MI de la nueva latente. Para ello la añadimos a currentDataForMI y currentMIsMatrix */
                estimateVariableMIs(currentMIsMatrix, data, currentDataForMI, currentModel, dag, newLatentVar);

            /* En caso negativo: paramos el bucle y eliminamos la nueva variable latente */
            } else {
                keepsImproving = false;

                dag.getParentSet(firstVar).removeParent(newLatentVar);
                dag.getParentSet(secondVar).removeParent(newLatentVar);
                dag.removeVariable(newLatentVar);

                variables.remove(newLatentVar);
            }
        }

        /* 4 - Aprendemos el modelo de forma global y devolvemos la solucion */
        VBEM_Global vbem__global = new VBEM_Global(this.finalVBEMConfig);
        vbem__global.learnModel(currentModel, dag);

        LogUtils.info("\nFinal score after global VBEM: " + currentModel.getLogProbabilityOfEvidence(), logLevel);

        return new Result(currentModel, currentModel.getLogProbabilityOfEvidence(), dag, "BLFM_BinG");
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

    // TODO: Estudiar como hacer para que itere unicamente por la triangular superior (dos iteradores con next())
    private Tuple2<Variable, Variable> highestMiVariables(Map<Variable, Map<Variable, Double>> misMatrix) {

        Variable bestX = null;
        Variable bestY = null;
        double bestValue = -1;

        List<Variable> keysList = new ArrayList<>(misMatrix.keySet());

        /* Iteramos por la triangular de la matriz de MIs para obtener el par de variables con valor maximo */
        for(int i = 0; i < keysList.size(); i++)
            for(int j = i+1; j < keysList.size(); j++){
                Variable x = keysList.get(i);
                Variable y = keysList.get(j);
                if(misMatrix.get(x).get(y) > bestValue) {
                    bestX = x;
                    bestY = y;
                    bestValue = misMatrix.get(x).get(y);
                }
            }

        return new Tuple2<>(bestX, bestY);
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

        /* Completamos los datos de la nueva variable latente y los almacenamos en currentData */
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

    private PlateuStructure greedyCardIncrease(Variable latentVar, PlateuStructure model, DAG dag) {
        PlateuStructure currentModel = model;
        double currentScore = model.getLogProbabilityOfEvidence();

        /* Mientras el score siga aumentando, seguimos incrementando la cardinalidad */
        while(true) {

            /* Incrementamos la cardinalidad de la variable */
            int newCardinality = latentVar.getNumberOfStates() + 1;
            latentVar.setNumberOfStates(newCardinality);
            latentVar.setStateSpaceType(new FiniteStateSpace(newCardinality));

            /* Creamos un nuevo Plateau para el aprendizaje donde omitimos copiar la variable en cuestion y sus hijos */
            HashSet<Variable> omittedVariables = new HashSet<>();
            omittedVariables.add(latentVar);
            omittedVariables.addAll(GraphUtilsAmidst.getChildren(latentVar, dag));
            PlateuStructure copyPlateauStructure = currentModel.deepCopy(dag, omittedVariables);

            /* Aprendemos el modelo de forma local */
            VBEM_Local vbem_local = new VBEM_Local();
            vbem_local.learnModel(copyPlateauStructure, dag, typeLocalVBEM.variablesToUpdate(latentVar, dag));

            /*
             * Comparamos el modelo generado con el mejor modelo actual.
             * - Si mejora el score se mantiene la cardinalidad y se repite el proceso
             * - Si no mejora el score, se resetea la cardinalidad y se devuelve el anterior modelo
             */
            if(vbem_local.getPlateuStructure().getLogProbabilityOfEvidence() > currentScore) {
                currentModel = vbem_local.getPlateuStructure();
                currentScore = vbem_local.getPlateuStructure().getLogProbabilityOfEvidence();
            } else {
                latentVar.setNumberOfStates(newCardinality - 1);
                latentVar.setStateSpaceType(new FiniteStateSpace(newCardinality - 1));
                return currentModel;
            }
        }
    }
}
