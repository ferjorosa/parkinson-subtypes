package eu.amidst.extension.learn.parameter.penalizer;

import eu.amidst.core.models.DAG;

public interface ElboPenalizer {

    double penalize(double elbo, DAG dag);
}
