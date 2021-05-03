package eu.amidst.extension.learn.structure.operator.incremental;

import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.extension.learn.structure.Result;
import eu.amidst.extension.util.tuple.Tuple3;

import java.util.PriorityQueue;
import java.util.Set;

public interface BlfmIncOperator {

    /**
     * Apply the operator to each of currentSet pairs of variables and return the highest scoring model.
     */
    Tuple3<Variable, Variable, Result> apply(Set<Variable> currentSet,
                                             PlateuStructure plateuStructure,
                                             DAG dag);

    /**
     * Apply the operator to each of the selected pairs of variables and return the highest scoring model.
     */
    Tuple3<Variable, Variable, Result> apply(PriorityQueue<Tuple3<Variable, Variable, Double>> selectedTriples,
                                             PlateuStructure plateuStructure,
                                             DAG dag);
}
