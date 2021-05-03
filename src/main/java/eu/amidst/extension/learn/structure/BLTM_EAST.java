package eu.amidst.extension.learn.structure;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.DAG;
import eu.amidst.extension.learn.parameter.VBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.VBEM_Global;
import eu.amidst.extension.learn.structure.operator.hc.tree.BltmHcOperator;
import eu.amidst.extension.util.LogUtils;

import java.util.Map;
import java.util.Set;

public class BLTM_EAST {

    private Set<BltmHcOperator> expansionOperators;

    private Set<BltmHcOperator> simplificationOperators;

    private Set<BltmHcOperator> adjustmentOperators;

    private int algorithmMaxIterations;

    private int processMaxIterations;

    private VBEMConfig initialVBEMConfig;

    private VBEMConfig processVBEMConfig;

    private boolean operatorGlobalVBEM;

    public BLTM_EAST(Set<BltmHcOperator> expansionOperators,
                     Set<BltmHcOperator> simplificationOperators,
                     Set<BltmHcOperator> adjustmentOperators,
                     boolean operatorGlobalVBEM){
        this(expansionOperators,
                simplificationOperators,
                adjustmentOperators,
                operatorGlobalVBEM,
                new VBEMConfig(),
                new VBEMConfig());
    }

    public BLTM_EAST(Set<BltmHcOperator> expansionOperators,
                     Set<BltmHcOperator> simplificationOperators,
                     Set<BltmHcOperator> adjustmentOperators,
                     boolean operatorGlobalVBEM,
                     VBEMConfig initialVBEMConfig,
                     VBEMConfig processVBEMConfig){

        this.expansionOperators = expansionOperators;
        this.simplificationOperators = simplificationOperators;
        this.adjustmentOperators = adjustmentOperators;

        this.algorithmMaxIterations = Integer.MAX_VALUE;
        this.processMaxIterations = Integer.MAX_VALUE;

        this.initialVBEMConfig = initialVBEMConfig;
        this.processVBEMConfig = processVBEMConfig;
        this.operatorGlobalVBEM = operatorGlobalVBEM;
    }

    public Result learnModel(DAG dag, DataOnMemory<DataInstance> data, Map<String, double[]> priors, boolean debug) {

        /* Realizamos un aprendizaje inicial de los parametros con la estructura argumento */
        VBEM initialVBEM = new VBEM(this.initialVBEMConfig);
        double previousIterationScore = initialVBEM.learnModel(data, dag, priors);
        Result bestResult = new Result(initialVBEM.getPlateuStructure(), previousIterationScore, dag, "Initial");

        LogUtils.printf("Initial score: " + previousIterationScore, debug);

        /* Bucle principal donde se llaman a los correspondientes procesos */
        int iterations = 0;
        while (iterations < this.algorithmMaxIterations) {
            iterations = iterations + 1;
            LogUtils.printf("\nBLTM_EAST ITERATION " + iterations + ":", debug);

            Result expansionResult = expansionProcess(bestResult, debug);
            Result adjustmentResult = adjustmentProcess(expansionResult, debug);
            Result simplificationResult = simplificationProcess(adjustmentResult, debug);


            if(simplificationResult.getElbo() <= bestResult.getElbo()){
                LogUtils.printf("\nEAST iteration doesn't improve the score" + simplificationResult.getElbo() + " <= " + bestResult.getElbo() + " (old best)", debug);
                LogUtils.printf("--------------------------------------------------", debug);
                return bestResult;
            }
            LogUtils.printf("\nEAST iteration improves the score" + simplificationResult.getElbo() + " > " + bestResult.getElbo() + " (old best)", debug);
            LogUtils.printf("--------------------------------------------------", debug);
            bestResult = simplificationResult;
        }

        return bestResult;
    }

    private Result expansionProcess(Result initialResult, boolean debug) {

        LogUtils.printf("\n-------------------", debug);
        LogUtils.printf("Expansion", debug);
        LogUtils.printf("-------------------", debug);

        Result bestResult = initialResult;
        int iterations = 0;
        while (iterations < this.processMaxIterations) {

            // Inicializamos con un Result falso
            Result bestIterationResult = new Result(null, -Double.MAX_VALUE, null, "NONE");

            iterations = iterations + 1;
            LogUtils.printf("\nIteration " + iterations + ":", debug);

            for (BltmHcOperator operator : this.expansionOperators) {
                Result result = operator.apply(bestResult.getPlateuStructure(), bestResult.getDag(), this.operatorGlobalVBEM);

                if (result.getElbo() == -Double.MAX_VALUE)
                    LogUtils.printf(result.getName() + " -> NONE", debug);
                else
                    LogUtils.printf(result.getName() + " -> " + result.getElbo(), debug);

                if (result.getElbo() > bestIterationResult.getElbo()) {
                    bestIterationResult = result;
                }
            }

            /* Una vez escogido el mejor modelo del proceso, si no hemos hecho aprendizaje global dentro de cada operador, lo hacemos ahora para ajustar los parametros */
            if(!this.operatorGlobalVBEM && bestIterationResult.getPlateuStructure() != null) { // Only if there is a change in the model
                VBEM_Global processVBEM = new VBEM_Global(this.processVBEMConfig);
                double score = processVBEM.learnModel(bestIterationResult.getPlateuStructure(), bestIterationResult.getDag());
                bestIterationResult = new Result(processVBEM.getPlateuStructure(), score, bestIterationResult.getDag(), bestIterationResult.getName());
                LogUtils.printf("\nAfter Global VBEM: " + score, debug);
            }

            /* En caso de que la iteracion no consiga mejorar el score del modelo, lo devolvemos */
            if(bestIterationResult.getElbo() <= bestResult.getElbo()) {
                LogUtils.printf("Doesn't improve the score: " + bestIterationResult.getElbo() + " <= " + bestResult.getElbo() + " (old best)", debug);
                return bestResult;
            } else {
                LogUtils.printf("Improves the score: " + bestIterationResult.getElbo() + " > " + bestResult.getElbo() + " (old best)", debug);
                bestResult = bestIterationResult;
            }
        }

        return bestResult;
    }

    private Result simplificationProcess(Result expansionResult, boolean debug) {

        LogUtils.printf("\n-------------------", debug);
        LogUtils.printf("Simplification", debug);
        LogUtils.printf("-------------------", debug);

        Result bestResult = expansionResult;
        int iterations = 0;
        while (iterations < this.processMaxIterations) {

            // Inicializamos con un Result falso
            Result bestIterationResult = new Result(null, -Double.MAX_VALUE, null, "NONE");

            iterations = iterations + 1;
            LogUtils.printf("\nIteration " + iterations + ":", debug);

            for (BltmHcOperator operator : this.simplificationOperators) {
                Result result = operator.apply(bestResult.getPlateuStructure(), bestResult.getDag(), this.operatorGlobalVBEM);

                if (result.getElbo() == -Double.MAX_VALUE)
                    LogUtils.printf(result.getName() + " -> NONE", debug);
                else
                    LogUtils.printf(result.getName() + " -> " + result.getElbo(), debug);

                if (result.getElbo() > bestIterationResult.getElbo()) {
                    bestIterationResult = result;
                }
            }

            /* Una vez escogido el mejor modelo del proceso, si no hemos hecho aprendizaje global dentro de cada operador, lo hacemos ahora para ajustar los parametros */
            if(!this.operatorGlobalVBEM && bestIterationResult.getPlateuStructure() != null) { // Only if there is a change in the model
                VBEM_Global processVBEM = new VBEM_Global(this.processVBEMConfig);
                double score = processVBEM.learnModel(bestIterationResult.getPlateuStructure(), bestIterationResult.getDag());
                bestIterationResult = new Result(processVBEM.getPlateuStructure(), score, bestIterationResult.getDag(), bestIterationResult.getName());
                LogUtils.printf("\nAfter Global VBEM: " + score, debug);
            }

            /* En caso de que la iteracion no consiga mejorar el score del modelo, lo devolvemos */
            if(bestIterationResult.getElbo() <= bestResult.getElbo()) {
                LogUtils.printf("Doesn't improve the score: " + bestIterationResult.getElbo() + " <= " + bestResult.getElbo() + " (old best)", debug);
                return bestResult;
            } else {
                LogUtils.printf("Improves the score: " + bestIterationResult.getElbo() + " > " + bestResult.getElbo() + " (old best)", debug);
                bestResult = bestIterationResult;
            }
        }

        return bestResult;
    }

    private Result adjustmentProcess(Result simplificationResult, boolean debug) {

        LogUtils.printf("\n-------------------", debug);
        LogUtils.printf("Adjustment", debug);
        LogUtils.printf("-------------------", debug);

        Result bestResult = simplificationResult;
        int iterations = 0;
        while (iterations < this.processMaxIterations) {

            // Inicializamos con un Result falso
            Result bestIterationResult = new Result(null, -Double.MAX_VALUE, null, "NONE");

            iterations = iterations + 1;
            LogUtils.printf("\nIteration " + iterations + ":", debug);

            for (BltmHcOperator operator : this.adjustmentOperators) {
                Result result = operator.apply(bestResult.getPlateuStructure(), bestResult.getDag(), this.operatorGlobalVBEM);

                if (result.getElbo() == -Double.MAX_VALUE)
                    LogUtils.printf(result.getName() + " -> NONE", debug);
                else
                    LogUtils.printf(result.getName() + " -> " + result.getElbo(), debug);

                if (result.getElbo() > bestIterationResult.getElbo()) {
                    bestIterationResult = result;
                }
            }

            /* Una vez escogido el mejor modelo del proceso, si no hemos hecho aprendizaje global dentro de cada operador, lo hacemos ahora para ajustar los parametros */
            if(!this.operatorGlobalVBEM && bestIterationResult.getPlateuStructure() != null) { // Only if there is a change in the model
                VBEM_Global processVBEM = new VBEM_Global(this.processVBEMConfig);
                double score = processVBEM.learnModel(bestIterationResult.getPlateuStructure(), bestIterationResult.getDag());
                bestIterationResult = new Result(processVBEM.getPlateuStructure(), score, bestIterationResult.getDag(), bestIterationResult.getName());
                LogUtils.printf("\nAfter Global VBEM: " + score, debug);
            }

            /* En caso de que la iteracion no consiga mejorar el score del modelo, lo devolvemos */
            if(bestIterationResult.getElbo() <= bestResult.getElbo()) {
                LogUtils.printf("Doesn't improve the score: " + bestIterationResult.getElbo() + " <= " + bestResult.getElbo() + " (old best)", debug);
                return bestResult;
            } else {
                LogUtils.printf("Improves the score: " + bestIterationResult.getElbo() + " > " + bestResult.getElbo() + " (old best)", debug);
                bestResult = bestIterationResult;
            }
        }

        return bestResult;
    }
}
