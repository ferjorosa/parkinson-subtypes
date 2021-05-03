package eu.amidst.extension.learn.parameter;

import eu.amidst.extension.learn.parameter.penalizer.BishopPenalizer;
import eu.amidst.extension.learn.parameter.penalizer.ElboPenalizer;

public class VBEMConfig {

    private long seed;
    private double threshold;
    private int maxIterations;
    private InitializationVBEM initialization;
    private ElboPenalizer elboPenalizer;

    public VBEMConfig() {
        this.seed = 0;
        this.threshold = 0.01;
        this.maxIterations = 100;
        this.initialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 10, 5, false);
        this.elboPenalizer = new BishopPenalizer();
    }

    public VBEMConfig(long seed,
                      double threshold,
                      int maxIterations,
                      InitializationVBEM initialization,
                      ElboPenalizer elboPenalizer) {
        this.seed = seed;
        this.threshold = threshold;
        this.maxIterations = maxIterations;
        this.initialization = initialization;
        this.elboPenalizer = elboPenalizer;
    }

    public long seed() { return seed; }

    public double threshold() { return threshold; }

    public int maxIterations() { return maxIterations;}

    public ElboPenalizer elboPenalizer() { return elboPenalizer; }

    public InitializationTypeVBEM initializationTypeVBEM() { return initialization.initializationType(); }

    public int initCandidates() { return initialization.nCandidates(); }

    public int initIterations() { return initialization.nIterations(); }

    public boolean initTestConvergence() { return initialization.testConvergence(); }
}
