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

import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;

// TODO: He implementado la nueva version del LocalEM, donde se actualiza el nuevo padre y el antiguo
public class BltmHcRelocateNode implements BltmHcOperator {

    private VBEMConfig localVBEMConfig;

    private VBEMConfig globalVBEMConfig;

    private TypeLocalVBEM typeLocalVBEM;

    public BltmHcRelocateNode(TypeLocalVBEM typeLocalVBEM) {
        this(new VBEMConfig(), new VBEMConfig(), typeLocalVBEM);
    }

    public BltmHcRelocateNode(VBEMConfig localVBEMConfig,
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
        Variable bestChild = null;
        Variable bestParent = null;
        Variable bestNewParent = null;

        Variables copyVariables = dag.getVariables().deepCopy();
        DAG copyDAG = dag.deepCopy(copyVariables);

        /* Obtenemos el conjunto de nodos latentes */
        List<Variable> latentVariables = copyVariables.getListOfVariables().stream()
                .filter(var->var.getAttribute() == null)
                .collect(Collectors.toList());

        /* Iteramos por el conjunto de nodos latentes y seleccionamos sus hijos y las otras variables latentes */
        for(Variable latentVariable: latentVariables) {

            List<Variable> otherLatentVariables = latentVariables.stream()
                    .filter(var -> !var.equals(latentVariable)).collect(Collectors.toList());

            List<Variable> observedChildren = GraphUtilsAmidst.getObservedChildren(latentVariable, copyDAG);

            /* Si el numero de hijos de esta variable es mayor que 2 y hay otras latentes, podriamos probar a trasladar uno a uno sus hijos */
            if(observedChildren.size() > 2 && otherLatentVariables.size() > 0) {

                /* Iteramos por los hijos */
                for (Variable child : observedChildren) {

                    /* Iteramos por las otras variables latentes y trasladamos la variable hija a su nueva particion */
                    for(Variable otherLatentVariable: otherLatentVariables) {

                        copyDAG.getParentSet(child).removeParent(latentVariable);
                        copyDAG.getParentSet(child).addParent(otherLatentVariable);

                        /* Creamos un nuevo Plateau para el aprendizaje donde omitimos el nuevo padre y sus hijos */
                        HashSet<Variable> omittedVariables = new HashSet<>();
                        omittedVariables.add(otherLatentVariable);
                        omittedVariables.addAll(GraphUtilsAmidst.getChildren(otherLatentVariable, copyDAG));
                        PlateuStructure copyPlateauStructure = plateuStructure.deepCopy(copyDAG, omittedVariables);

                        /* Aprendemos el modelo de forma local */
                        VBEM_Local localVBEM = new VBEM_Local(this.localVBEMConfig);
                        localVBEM.learnModel(copyPlateauStructure, copyDAG, typeLocalVBEM.variablesToUpdate(latentVariable, otherLatentVariable, copyDAG));

                        /* Comparamos el modelo generado con el mejor modelo actual */
                        if(localVBEM.getPlateuStructure().getLogProbabilityOfEvidence() > bestModelScore) {
                            bestModel = localVBEM.getPlateuStructure();
                            bestModelScore = localVBEM.getPlateuStructure().getLogProbabilityOfEvidence();
                            bestChild = child;
                            bestParent = latentVariable;
                            bestNewParent = otherLatentVariable;
                        }

                        /* Modificamos el grafo y devolvemos la variable a su posicion inicial para resetear el proceso */
                        copyDAG.getParentSet(child).removeParent(otherLatentVariable);
                        copyDAG.getParentSet(child).addParent(latentVariable);
                    }
                }
            }
        }

        /* Si el operador produjo un mejor modelo, lo aprendemos con VBEM_HC de forma "global" */
        if(bestModelScore > -Double.MAX_VALUE) {

            // Modificamos la estructura para que no haya diferencias con el PlateauStructure
            copyDAG.getParentSet(bestChild).removeParent(bestParent);
            copyDAG.getParentSet(bestChild).addParent(bestNewParent);

            if(doGlobalVBEM) {
                VBEM_Global globalVBEM = new VBEM_Global(this.globalVBEMConfig);
                globalVBEM.learnModel(bestModel, copyDAG);

                bestModel = globalVBEM.getPlateuStructure();
                bestModelScore = globalVBEM.getPlateuStructure().getLogProbabilityOfEvidence();
            }
        }

        /* Devolvemos el resultado */
        return new Result(bestModel, bestModelScore, copyDAG, "RelocateNode");
    }
}
