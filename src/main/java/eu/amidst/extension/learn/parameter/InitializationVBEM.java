package eu.amidst.extension.learn.parameter;

public class InitializationVBEM {

    private InitializationTypeVBEM initializationType;

    private int nCandidates;

    private int nIterations;

    private boolean testConvergence;

    public InitializationVBEM(InitializationTypeVBEM initializationType, int nCandidates, int nIterations, boolean testConvergence) {
        this.initializationType = initializationType;
        this.nCandidates = nCandidates;
        this.nIterations = nIterations;
        this.testConvergence = testConvergence;
    }

    public InitializationTypeVBEM initializationType() { return this.initializationType; }

    public int nCandidates() { return nCandidates; }

    public int nIterations() { return nIterations; }

    public boolean testConvergence() { return testConvergence; }
}
