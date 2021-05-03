/**
 * 
 */
package org.latlab.core.reasoner;

import org.latlab.core.model.BayesNet;

/**
 * Represents a impossible is used in propagation.
 * 
 * @author leonard
 * 
 */
public class ImpossibleEvidenceException extends RuntimeException {
    private static final long serialVersionUID = 4130276544022141910L;

    private final Evidences evidences;
    private final BayesNet model;

    public ImpossibleEvidenceException(BayesNet model, Evidences evidences) {
        this.evidences = evidences;
        this.model = model;
    }

    /**
     * Returns the evidences that lead to a improper propagation.
     * 
     * @return evidences that lead to a improper propagation
     */
    public Evidences evidences() {
        return evidences;
    }
    
    /**
     * Returns the model on which the propagation is performed on.
     * 
     * @return model on which the propagation is performed on
     */
    public BayesNet model() {
       return model; 
    }
}
