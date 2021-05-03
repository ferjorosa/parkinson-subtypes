package org.latlab.core.learner.geast;

@SuppressWarnings("serial")
public class EstimationNaNException extends RuntimeException {
    private final Estimation estimation;

    public EstimationNaNException(Estimation estimation) {
        this.estimation = estimation;
    }

    public EstimationNaNException(String message, Estimation estimation) {
        super(message);
        this.estimation = estimation;
    }
    
    public Estimation estimation() {
        return estimation;
    }
}
