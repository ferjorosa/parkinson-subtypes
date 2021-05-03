package eu.amidst.extension.learn.structure;

import eu.amidst.core.learning.parametric.bayesian.utils.PlateuStructure;
import eu.amidst.core.models.DAG;

public class Result {

    private PlateuStructure posteriorPlateuStructure;

    private double elbo;

    private DAG dag;

    private String name;

    public Result(PlateuStructure posteriorPlateuStructure, double elbo, DAG dag, String name) {
        this.posteriorPlateuStructure = posteriorPlateuStructure;
        this.elbo = elbo;
        this.dag = dag;
        this.name = name;
    }

    public PlateuStructure getPlateuStructure() {
        return posteriorPlateuStructure;
    }

    public double getElbo() {
        return elbo;
    }

    public DAG getDag() {
        return dag;
    }

    public String getName() {
        return name;
    }
}
