package eu.amidst.extension.learn.structure.hillclimber.operator;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHcOperation;
import eu.amidst.extension.util.tuple.Tuple2;

import java.util.Map;
import java.util.Set;

public interface BayesianHcOperator {

    BayesianHcOperation apply(DAG dag,
                              DataOnMemory<DataInstance> data,
                              Map<String, double[]> priorsParameters,
                              Map<Tuple2<Variable, Set<Variable>>, Double> scores);
}
