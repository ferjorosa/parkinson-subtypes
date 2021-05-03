package eu.amidst.extension.learn.structure.glsl.operator;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.tuple.Tuple3;

public interface GLSL_Operator {

    Tuple3<String, BayesianNetwork, Double> apply(DAG dag, DataOnMemory<DataInstance> data, LogUtils.LogLevel logLevel);
}
