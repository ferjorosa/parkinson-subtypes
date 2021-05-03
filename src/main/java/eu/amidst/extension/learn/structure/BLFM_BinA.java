package eu.amidst.extension.learn.structure;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.core.variables.stateSpaceTypes.FiniteStateSpace;
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
import java.util.stream.Collectors;

public class BLFM_BinA {

    public enum LinkageType {
        SINGLE,
        COMPLETE,
        AVERAGE
    }

    private LinkageType linkageType;

    private int n_neighbors_mi;

    private DistanceFunction distanceFunction_mi;

    private boolean gaussianNoise_mi;

    private long gaussianNoiseSeed;

    private boolean normalizedMI;

    private VBEMConfig initialVBEMConfig;

    private VBEMConfig localVBEMConfig;

    private VBEMConfig finalVBEMConfig;

    private TypeLocalVBEM typeLocalVBEM;

    private int latentVarNameCounter = 0;

    public BLFM_BinA(int n_neighbors_mi,
                     DistanceFunction distanceFunction_mi,
                     boolean gaussianNoise_mi,
                     long gaussianNoiseSeed,
                     boolean normalizedMI,
                     LinkageType linkageType,
                     TypeLocalVBEM typeLocalVBEM) {
        this(n_neighbors_mi,
                distanceFunction_mi,
                gaussianNoise_mi,
                gaussianNoiseSeed,
                normalizedMI,
                linkageType,
                new VBEMConfig(),
                new VBEMConfig(),
                new VBEMConfig(),
                typeLocalVBEM);
    }

    public BLFM_BinA(int n_neighbors_mi,
                     DistanceFunction distanceFunction_mi,
                     boolean gaussianNoise_mi,
                     long gaussianNoiseSeed,
                     boolean normalizedMI,
                     LinkageType linkageType,
                     VBEMConfig initialVBEMConfig,
                     VBEMConfig localVBEMConfig,
                     VBEMConfig finalVBEMConfig,
                     TypeLocalVBEM typeLocalVBEM) {
        this.n_neighbors_mi = n_neighbors_mi;
        this.distanceFunction_mi = distanceFunction_mi;
        this.gaussianNoise_mi = gaussianNoise_mi;
        this.gaussianNoiseSeed = gaussianNoiseSeed;
        this.normalizedMI = normalizedMI;
        this.linkageType = linkageType;
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

        /* 1 - Estimamos la MI entre cada par de atributos de los datos */
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

        LogUtils.info("Initial score: " + currentModel.getLogProbabilityOfEvidence(), logLevel);

        /* 2 - Bucle principal */
        boolean keepsImproving = true;
        int iteration = 0;
        while(keepsImproving && currentSet.size() > 1) {

            iteration++;

            /* 3 - Escogemos el par de variables con mayor MI */
            Tuple2<Variable, Variable> highestMiVariablesIndexes = highestMiVariables(currentSet, currentMIsMatrix, dag);
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
                currentSet.remove(firstVar);
                currentSet.remove(secondVar);
                currentSet.add(newLatentVar);

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
        VBEM_Global vbem_global = new VBEM_Global(this.finalVBEMConfig);
        vbem_global.learnModel(currentModel, dag);

        LogUtils.info("\nFinal score after global VBEM: " + currentModel.getLogProbabilityOfEvidence(), logLevel);

        return new Result(currentModel, currentModel.getLogProbabilityOfEvidence(), dag, "BLFM_BinA");
    }

    /** En este caso necesitamos currentSet porque no se eliminan los valores de currentMIsMatrix */
    private Tuple2<Variable, Variable> highestMiVariables(Set<Variable> currentSet, Map<Variable, Map<Variable, Double>> currentMIsMatrix, DAG dag) {

        Variable bestX = null;
        Variable bestY = null;
        double bestMI = -1;

        List<Variable> currentSetInListForm = new ArrayList<>(currentSet);

        /* Iteramos por la triangular de la matriz de MIs para obtener el par de variables con valor maximo */
        for(int i = 0; i < currentSetInListForm.size(); i++)
            for(int j = i+1; j < currentSetInListForm.size(); j++){
                Variable x = currentSetInListForm.get(i);
                Variable y = currentSetInListForm.get(j);

                /* Realizamos DFS para almacenar las variables hijas de X e Y*/
                Map<Variable, Integer> visitedNodesX = new HashMap<>();
                dag.dfs(x, visitedNodesX);
                Map<Variable, Integer> visitedNodesY = new HashMap<>();
                dag.dfs(y, visitedNodesY);

                /* Seleccionamos las variables observadas de cada una de ellas */
                List<Variable> xLeafs = visitedNodesX.keySet().stream().filter(Variable::isObservable).collect(Collectors.toList());
                List<Variable> yLeafs = visitedNodesY.keySet().stream().filter(Variable::isObservable).collect(Collectors.toList());

                /* Estimamos la MI y almacenamos las variables si mejora el valor actual */
                double mi = estimateAgglomerativeMI(xLeafs, yLeafs, currentMIsMatrix);
                if(mi > bestMI) {
                    bestX = x;
                    bestY = y;
                    bestMI = mi;
                }
            }
        return new Tuple2<>(bestX, bestY);
    }

    private double estimateAgglomerativeMI(List<Variable> xLeafs, List<Variable> yLeafs, Map<Variable, Map<Variable, Double>> currentMIsMatrix) {

        double returnMI = 0;

        switch (this.linkageType) {

            case SINGLE:
                double maxMI = -1;
                for(Variable xLeaf: xLeafs)
                    for(Variable yLeaf: yLeafs) {
                        double mi = currentMIsMatrix.get(xLeaf).get(yLeaf);
                        if(mi > maxMI)
                            maxMI = mi;
                    }
                returnMI = maxMI;
                break;

            case AVERAGE:
                double averageMI = 0;
                int N = (xLeafs.size() + yLeafs.size()) * (xLeafs.size() + yLeafs.size());
                for(Variable xLeaf: xLeafs)
                    for(Variable yLeaf: yLeafs) {
                        double mi = currentMIsMatrix.get(xLeaf).get(yLeaf);
                        averageMI += mi / N;
                    }
                returnMI = averageMI;
                break;

            case COMPLETE:
                double minMI = Double.MAX_VALUE;
                for(Variable xLeaf: xLeafs)
                    for(Variable yLeaf: yLeafs) {
                        double mi = currentMIsMatrix.get(xLeaf).get(yLeaf);
                        if(mi < minMI)
                            minMI = mi;
                    }
                returnMI = minMI;
                break;
        }
        return returnMI;
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
