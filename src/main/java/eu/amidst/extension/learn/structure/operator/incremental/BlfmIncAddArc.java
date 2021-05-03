package eu.amidst.extension.learn.structure.operator.incremental;

import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.VBEM_Local;
import eu.amidst.extension.learn.structure.Result;
import eu.amidst.extension.learn.structure.typelocalvbem.TypeLocalVBEM;
import eu.amidst.extension.util.tuple.Tuple2;
import eu.amidst.extension.util.tuple.Tuple3;

import java.util.*;

public class BlfmIncAddArc implements BlfmIncOperator {

    private boolean allowObservedToObservedDiscreteArc;

    private boolean allowObservedToObservedArc;

    private boolean allowObservedToLatentArc;

    private VBEMConfig localVBEMConfig;

    private TypeLocalVBEM typeLocalVBEM;

    public BlfmIncAddArc(boolean allowObservedToObservedArc,
                         boolean allowObservedToLatentArc,
                         boolean allowObservedToObservedDiscreteArc,
                         TypeLocalVBEM typeLocalVBEM) {
        this(allowObservedToObservedArc, allowObservedToLatentArc, allowObservedToObservedDiscreteArc, new VBEMConfig(), typeLocalVBEM);
    }

    public BlfmIncAddArc(boolean allowObservedToObservedArc,
                         boolean allowObservedToLatentArc,
                         boolean allowObservedToObservedDiscreteArc,
                         VBEMConfig localVBEMConfig,
                         TypeLocalVBEM typeLocalVBEM) {
        this.allowObservedToObservedArc = allowObservedToObservedArc;
        this.allowObservedToLatentArc = allowObservedToLatentArc;
        this.allowObservedToObservedDiscreteArc = allowObservedToObservedDiscreteArc;
        this.localVBEMConfig = localVBEMConfig;
        this.typeLocalVBEM = typeLocalVBEM;
    }

    @Override
    public Tuple3<Variable, Variable, Result> apply(Set<Variable> currentSet, PlateuStructure plateuStructure, DAG dag) {

        PlateuStructure bestModel = plateuStructure;
        double bestModelScore = -Double.MAX_VALUE;
        Tuple2<Variable, Variable> bestPair = null;

        /* Create a copy of the variables and DAG objects */
        Variables copyVariables = dag.getVariables().deepCopy();
        DAG copyDAG = dag.deepCopy(copyVariables);
        Set<Variable> copyCurrentSet = new LinkedHashSet<>();
        for(Variable var: currentSet)
            copyCurrentSet.add(copyVariables.getVariableByName(var.getName()));

        /*
         * Iterate through all the combinations of variables in the currentSet. We consider all combinations becase we are
         * searching for the best directed arc.
         */
        for(Variable fromVar: copyCurrentSet) {
            for (Variable toVar : copyCurrentSet) {
                // We dont allow arcs between a variable and itself.
                // We dont also allow arcs from continuous to discrete.
                if (!fromVar.equals(toVar) && !(fromVar.isContinuous() && toVar.isDiscrete())) {
                    if ((!fromVar.isObservable() && !toVar.isObservable())                                           // LV -> LV
                            || (!fromVar.isObservable() && toVar.isObservable())                                     // LV -> OV
                            || (fromVar.isObservable() && !toVar.isObservable() && this.allowObservedToLatentArc)    // OV -> LV        [only if allowed]
                            || (fromVar.isObservable() && toVar.isObservable() && this.allowObservedToObservedArc))  // OV -> OV        [only if allowed]
                    {
                        if(fromVar.isObservable() && toVar.isObservable() && fromVar.isDiscrete() && toVar.isDiscrete() && !this.allowObservedToObservedDiscreteArc)
                            continue;

                        copyDAG.getParentSet(toVar).addParent(fromVar);

                        /* Create a new plateau by copying current one and omitting the var receiving the arc) */
                        HashSet<Variable> omittedVariables = new HashSet<>();
                        omittedVariables.add(toVar);
                        PlateuStructure copyPlateauStructure = plateuStructure.deepCopy(copyDAG, omittedVariables);

                        /* Aprendemos el modelo de forma local, actualizando ambas variables (con sus hijos) de forma local */
                        VBEM_Local localVBEM = new VBEM_Local(this.localVBEMConfig);
                        localVBEM.learnModel(copyPlateauStructure, copyDAG, typeLocalVBEM.variablesToUpdate(fromVar, toVar, copyDAG));

                        /* Compare its score with current best model */
                        if (localVBEM.getPlateuStructure().getLogProbabilityOfEvidence() > bestModelScore) {
                            bestModel = localVBEM.getPlateuStructure();
                            bestModelScore = localVBEM.getPlateuStructure().getLogProbabilityOfEvidence();
                            bestPair = new Tuple2<>(fromVar, toVar);
                        }

                        /* Remove the newly created arc to reset the process for the next pair */
                        copyDAG.getParentSet(toVar).removeParent(fromVar);
                    }
                }
            }
        }

        /* Modify the DAG with the best arc */
        if(bestModelScore > -Double.MAX_VALUE) {
            copyDAG.getParentSet(bestPair.getSecond()).addParent(bestPair.getFirst());
            return new Tuple3<>(bestPair.getFirst(), bestPair.getSecond(), new Result(bestModel, bestModelScore, copyDAG, "AddArc"));
        }

        return new Tuple3<>(null, null, new Result(bestModel, bestModelScore, copyDAG, "AddArc"));
    }

    @Override
    public Tuple3<Variable, Variable, Result> apply(PriorityQueue<Tuple3<Variable, Variable, Double>> selectedTriples, PlateuStructure plateuStructure, DAG dag) {

        PlateuStructure bestModel = plateuStructure;
        double bestModelScore = -Double.MAX_VALUE;
        Tuple2<Variable, Variable> bestPair = null;

        /* Create a copy of the variables and DAG objects */
        Variables copyVariables = dag.getVariables().deepCopy();
        DAG copyDAG = dag.deepCopy(copyVariables);

        /* Iterate through the combinations of selected triples */
        for(Tuple3<Variable, Variable, Double> triple: selectedTriples){

            /* Create a tuple with the variables' copies as items */
            List<Variable> tupleList = new ArrayList<>(2);
            tupleList.add(copyVariables.getVariableByName(triple.getFirst().getName())); // variable copy
            tupleList.add(copyVariables.getVariableByName(triple.getSecond().getName())); // variable copy

            for(Variable fromVar: tupleList) {
                for (Variable toVar : tupleList) {
                    // We dont allow arcs between a variable and itself.
                    // We dont also allow arcs from continuous to discrete.
                    if (!fromVar.equals(toVar) && !(fromVar.isContinuous() && toVar.isDiscrete())) {
                        if ((!fromVar.isObservable() && !toVar.isObservable())                                           // LV -> LV
                                || (!fromVar.isObservable() && toVar.isObservable())                                     // LV -> OV
                                || (fromVar.isObservable() && !toVar.isObservable() && this.allowObservedToLatentArc)    // OV -> LV        [only if allowed]
                                || (fromVar.isObservable() && toVar.isObservable() && this.allowObservedToObservedArc))  // OV -> OV        [only if allowed]
                        {
                            if(fromVar.isObservable() && toVar.isObservable() && fromVar.isDiscrete() && toVar.isDiscrete() && !this.allowObservedToObservedDiscreteArc)
                                continue;

                            copyDAG.getParentSet(toVar).addParent(fromVar);

                            /* Create a new plateau by copying current one and omitting the var receiving the arc) */
                            HashSet<Variable> omittedVariables = new HashSet<>();
                            omittedVariables.add(toVar);
                            PlateuStructure copyPlateauStructure = plateuStructure.deepCopy(copyDAG, omittedVariables);

                            /* Aprendemos el modelo de forma local, actualizando ambas variables (con sus hijos) de forma local */
                            VBEM_Local localVBEM = new VBEM_Local(this.localVBEMConfig);
                            localVBEM.learnModel(copyPlateauStructure, copyDAG, typeLocalVBEM.variablesToUpdate(fromVar, toVar, copyDAG));

                            /* Compare its score with current best model */
                            if (localVBEM.getPlateuStructure().getLogProbabilityOfEvidence() > bestModelScore) {
                                bestModel = localVBEM.getPlateuStructure();
                                bestModelScore = localVBEM.getPlateuStructure().getLogProbabilityOfEvidence();
                                bestPair = new Tuple2<>(fromVar, toVar);
                            }

                            /* Remove the newly created arc to reset the process for the next pair */
                            copyDAG.getParentSet(toVar).removeParent(fromVar);
                        }
                    }
                }
            }
        }

        /* Modify the DAG with the best arc */
        if(bestModelScore > -Double.MAX_VALUE) {
            copyDAG.getParentSet(bestPair.getSecond()).addParent(bestPair.getFirst());
            return new Tuple3<>(bestPair.getFirst(), bestPair.getSecond(), new Result(bestModel, bestModelScore, copyDAG, "AddArc"));
        }

        return new Tuple3<>(null, null, new Result(bestModel, bestModelScore, copyDAG, "AddArc"));
    }
}
