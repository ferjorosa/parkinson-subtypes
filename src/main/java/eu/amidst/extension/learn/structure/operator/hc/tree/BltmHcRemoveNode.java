package eu.amidst.extension.learn.structure.operator.hc.tree;

import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.VBEM_Global;
import eu.amidst.extension.learn.parameter.VBEM_Local;
import eu.amidst.extension.learn.structure.Result;
import eu.amidst.extension.learn.structure.typelocalvbem.TypeLocalVBEM;
import eu.amidst.extension.util.GraphUtilsAmidst;
import eu.amidst.extension.util.tuple.Tuple2;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

public class BltmHcRemoveNode implements BltmHcOperator {

    private VBEMConfig localVBEMConfig;

    private VBEMConfig globalVBEMConfig;

    private TypeLocalVBEM typeLocalVBEM;

    public BltmHcRemoveNode(TypeLocalVBEM typeLocalVBEM) {
        this(new VBEMConfig(), new VBEMConfig(), typeLocalVBEM);
    }

    public BltmHcRemoveNode(VBEMConfig localVBEMConfig,
                            VBEMConfig globalVBEMConfig,
                            TypeLocalVBEM typeLocalVBEM) {
        this.localVBEMConfig = localVBEMConfig;
        this.globalVBEMConfig = globalVBEMConfig;
        this.typeLocalVBEM = typeLocalVBEM;
    }

    @Override
    public Result apply(PlateuStructure plateuStructure, DAG dag, boolean doGlobalVBEM) {

        PlateuStructure bestModel = plateuStructure;
        double bestModelScore = -Double.MAX_VALUE;
        Tuple2<Variable, Variable> bestLatentPair = null;

        Variables copyVariables = dag.getVariables().deepCopy();
        DAG copyDAG = dag.deepCopy(copyVariables);

        /* Obtenemos el conjunto de nodos latentes */
        List<Variable> latentVariables = copyVariables.getListOfVariables().stream()
                .filter(var->var.getAttribute() == null)
                .collect(Collectors.toList());

        /* Seleccionamos aquellos nodos latentes que tengan un padre latente y generamos una Tupla */
        List<Tuple2<Variable, Variable>> latentVarsWithLatentParent = new ArrayList<>();
        for(Variable latentVariable: latentVariables){

            Optional<Variable> latentParent = copyDAG.getParentSet(latentVariable).getParents().stream()
                    .filter(latentVariables::contains)
                    .findFirst();

            if(latentParent.isPresent())
                latentVarsWithLatentParent.add(new Tuple2<>(latentVariable, latentParent.get()));
        }

        /* Iteramos por los pares de variables donde la variable a eliminar es la primera y el padre la segunda */
        for(Tuple2<Variable, Variable> latentPair: latentVarsWithLatentParent) {

            /* Eliminamos la variable latente y añadimos sus hijos correspondientes al padre */
            Variable latentVariable = latentPair.getFirst();
            Variable latentParent = latentPair.getSecond();
            List<Variable> latentVarChildren = GraphUtilsAmidst.getChildren(latentVariable, copyDAG);

            for(Variable child: latentVarChildren) {
                copyDAG.getParentSet(child).removeParent(latentVariable);
                copyDAG.getParentSet(child).addParent(latentParent);
            }

            copyDAG.removeVariable(latentVariable);
            copyVariables.remove(latentVariable);

            /* Creamos un nuevo Plateau para el aprendizaje donde omitimos la variable eliminada, sus hijas y su nuevo padre */
            HashSet<Variable> omittedVariables = new HashSet<>();
            omittedVariables.add(latentVariable);
            omittedVariables.add(latentParent);
            omittedVariables.addAll(GraphUtilsAmidst.getChildren(latentParent, copyDAG));
            PlateuStructure copyPlateauStructure = plateuStructure.deepCopy(copyDAG, omittedVariables);

            /* Aprendemos el modelo de forma local */
            VBEM_Local localVBEM = new VBEM_Local(this.localVBEMConfig);
            localVBEM.learnModel(copyPlateauStructure, copyDAG, typeLocalVBEM.variablesToUpdate(latentParent, copyDAG));

            /* Comparamos el modelo generado con el mejor modelo actual */
            if(localVBEM.getPlateuStructure().getLogProbabilityOfEvidence() > bestModelScore) {
                bestModel = localVBEM.getPlateuStructure();
                bestModelScore = localVBEM.getPlateuStructure().getLogProbabilityOfEvidence();
                bestLatentPair = latentPair;
            }

            /* Modificamos el grafo y volvemos a añadir el nodo con sus hijos para resetear el proceso */
            copyVariables.add(latentVariable);
            copyDAG.addVariable(latentVariable);

            copyDAG.getParentSet(latentVariable).addParent(latentParent);
            for(Variable child: latentVarChildren) {
                copyDAG.getParentSet(child).removeParent(latentParent);
                copyDAG.getParentSet(child).addParent(latentVariable);
            }
        }

        /* Si el operador produjo un mejor modelo, lo aprendemos con VBEM_HC de forma "global" */
        // TODO: Escoger el tipo de inicializacion a utilizar aqui, podria ser incluso NONE
        if(bestModelScore > -Double.MAX_VALUE) {

            // Modificamos el grafo para que no haya diferencias con la estructura del Plateau
            Variable latentVariable = bestLatentPair.getFirst();
            Variable latentParent = bestLatentPair.getSecond();
            List<Variable> latentVarChildren = GraphUtilsAmidst.getChildren(latentVariable, copyDAG);

            for(Variable child: latentVarChildren) {
                copyDAG.getParentSet(child).removeParent(latentVariable);
                copyDAG.getParentSet(child).addParent(latentParent);
            }

            copyDAG.removeVariable(latentVariable);
            copyVariables.remove(latentVariable);

            if(doGlobalVBEM) {
                VBEM_Global globalVBEM = new VBEM_Global(this.globalVBEMConfig);
                globalVBEM.learnModel(bestModel, copyDAG);

                bestModel = globalVBEM.getPlateuStructure();
                bestModelScore = globalVBEM.getPlateuStructure().getLogProbabilityOfEvidence();
            }
        }

        /* Devolvemos el resultado */
        return new Result(bestModel, bestModelScore, copyDAG, "RemoveNode");
    }
}
