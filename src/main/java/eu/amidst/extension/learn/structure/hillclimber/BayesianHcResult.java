package eu.amidst.extension.learn.structure.hillclimber;

import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;

public class BayesianHcResult {

    private double elbo;

    private BayesianNetwork bayesianNetwork;

    private String name;

    public BayesianHcResult(double elbo, BayesianNetwork bayesianNetwork, String name) {
        this.elbo = elbo;
        this.bayesianNetwork = bayesianNetwork;
        this.name = name;
    }

    public double getElbo() {
        return elbo;
    }

    public BayesianNetwork getBayesianNetwork() {
        return this.bayesianNetwork;
    }

    public DAG getDag() {
        return this.bayesianNetwork.getDAG();
    }

    public String getName() {
        return name;
    }
}
