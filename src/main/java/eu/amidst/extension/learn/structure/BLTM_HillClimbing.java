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

/**
 * TODO: Explicar que segun si permitimos o no el operatorGlobalVBEM estariamos ante dos algoritmos diferentes:
 * Sin: Clasico HSHC de Zhang & Kocka
 * Con: Algoritmo Hill-climbing mas tradicional
 */
public class BLTM_HillClimbing {

    private Set<BltmHcOperator> operators;

    private boolean operatorGlobalVBEM;

    private int maxIterations;

    private VBEMConfig initialVBEMConfig;

    private VBEMConfig iterationVBEMConfig;

    public BLTM_HillClimbing(Set<BltmHcOperator> operators, boolean operatorGlobalVBEM) {
        this(operators, operatorGlobalVBEM, new VBEMConfig(), new VBEMConfig());
    }

    public BLTM_HillClimbing(Set<BltmHcOperator> operators,
                             boolean operatorGlobalVBEM,
                             VBEMConfig initialVBEMConfig,
                             VBEMConfig iterationVBEMConfig) {
        this.operators = operators;
        this.maxIterations = Integer.MAX_VALUE;
        this.operatorGlobalVBEM = operatorGlobalVBEM;
        this.initialVBEMConfig = initialVBEMConfig;
        this.iterationVBEMConfig = iterationVBEMConfig;
    }

    // TODO: Empezamos sin permitir que se establezcan priors. Podriamos hacer que las nuevas LVs siempre tengan una prior por defecto
    public Result learnModel(DAG dag, DataOnMemory<DataInstance> data, Map<String, double[]> priors, boolean debug) {

        /* Realizamos un aprendizaje inicial de los parametros con la estructura argumento */
        VBEM initialVBEM = new VBEM(this.initialVBEMConfig);
        double previousIterationScore = initialVBEM.learnModel(data, dag, priors);
        Result bestResult = new Result(initialVBEM.getPlateuStructure(), previousIterationScore, dag, "Initial");

        LogUtils.printf("Initial score: " + previousIterationScore, debug);

        /* Bucle principal. Iteramos por los diferentes operadores, seleccionamos el mejor y comparamos su score con el modelo previo */
        int iterations = 0;
        while (iterations < this.maxIterations) {
            iterations = iterations + 1;
            LogUtils.printf("\nIteration " + iterations + ":", debug);

            // Inicializamos con un Result falso
            Result bestIterationResult = new Result(null, -Double.MAX_VALUE, null, "NONE");

            for (BltmHcOperator operator : this.operators) {
                Result result = operator.apply(bestResult.getPlateuStructure(), bestResult.getDag(), this.operatorGlobalVBEM);

                if(result.getElbo() == -Double.MAX_VALUE)
                    LogUtils.printf(result.getName() + " -> NONE", debug);
                else
                    LogUtils.printf(result.getName() + " -> " + result.getElbo(), debug);

                if(result.getElbo() > bestIterationResult.getElbo()) {
                    bestIterationResult = result;
                }
            }

            LogUtils.printf("\nBest operator: "+bestIterationResult.getName(), debug);

            /* Una vez escogido el mejor modelo del proceso, si no hemos hecho aprendizaje global dentro de cada operador, lo hacemos ahora para ajustar los parametros */
            if(!this.operatorGlobalVBEM && bestIterationResult.getPlateuStructure() != null) { // Only if there is a change in the model
                VBEM_Global iterationVBEM = new VBEM_Global(this.iterationVBEMConfig);
                double score = iterationVBEM.learnModel(bestIterationResult.getPlateuStructure(), bestIterationResult.getDag());
                bestIterationResult = new Result(iterationVBEM.getPlateuStructure(), score, bestIterationResult.getDag(), bestIterationResult.getName());
                LogUtils.printf("\nAfter Global VBEM: " + score, debug);
            }

            /* En caso de que la iteracion no consiga mejorar el score del modelo, lo devolvemos */
            if(bestIterationResult.getElbo() <= bestResult.getElbo()) {
                LogUtils.printf("Doesn't improve the score: " + bestIterationResult.getElbo() + " <= " + bestResult.getElbo() + " (old best)", debug);
                LogUtils.printf("--------------------------------------------------", debug);
                return bestResult;
            }

            LogUtils.printf("Improves the score: " + bestIterationResult.getElbo() + " > " + bestResult.getElbo() + " (old best)", debug);
            LogUtils.printf("--------------------------------------------------", debug);
            bestResult = bestIterationResult;
        }

        /*
         * En caso de que el numero de iteraciones se haya superado mientras el modelo mejoraba de forma satisfactoria,
         * se devuelve el resultado de la ultima iteracion realizada.
         */
        return bestResult;
    }
}
