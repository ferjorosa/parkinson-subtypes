package eu.amidst.extension.learn.parameter.penalizer;

import eu.amidst.core.models.DAG;

public class NoPenalizer implements ElboPenalizer {

    @Override
    public double penalize(double elbo, DAG dag) {
        return elbo;
    }
}
