package eu.amidst.extension.learn.structure.vbsem;

public class InitializationVBSEM {

    private InitializationTypeVBSEM initializationType;

    private int nCandidates;

    private int nIterations;

    private double sparsityCoefficient;

    private boolean testConvergence;

    public InitializationVBSEM(InitializationTypeVBSEM initializationType,
                               int nCandidates,
                               int nIterations,
                               double sparsityCoefficient,
                               boolean testConvergence) {
        this.initializationType = initializationType;
        this.nCandidates = nCandidates;
        this.nIterations = nIterations;
        this.sparsityCoefficient = sparsityCoefficient;
        this.testConvergence = testConvergence;
    }

    public InitializationTypeVBSEM initializationType() { return this.initializationType; }

    public int nCandidates() { return nCandidates; }

    public int nIterations() { return nIterations; }

    public double sparsityCoefficient() {
        return sparsityCoefficient;
    }

    public boolean testConvergence() { return testConvergence; }
}
