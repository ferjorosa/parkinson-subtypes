/**
 * 
 */
package org.latlab.core.learner.geast;

/**
 * @author leonard
 * 
 */
public class EmParameters {

    public int restarts = 64;

    public boolean reuseParameters = true;

    public int initialIterations = 1;

    public int maxSteps = 500;

    public int secondStageSteps = 0;

    public double threshold = 1e-4;

    public double smoothing = 0;

    /**
     * Sometimes the random parameter may lead to a diverge estimation in the
     * EM. This parameter sets the minimum of restarts it should try to avoid a
     * {@code Double.NaN} for a estimation generated in the initial stage.
     */
    public int minimumRetryForNaN = 16;

    public EmParameters() {

    }

    public EmParameters(
        boolean reuseParameters, int restarts, int secondStageSteps,
        int maxSteps, double threshold) {
        this.reuseParameters = reuseParameters;
        this.restarts = restarts;
        this.secondStageSteps = secondStageSteps;
        this.maxSteps = maxSteps;
        this.threshold = threshold;
    }

    public String xmlAttributes() {
        return String.format(
            "reuse='%s' restarts='%d' maxSteps='%d' secondStageSteps='%d' "
                + "threshold='%.2e' initial='%d' minForNaN='%d'",
            reuseParameters, restarts, maxSteps, secondStageSteps, threshold,
            initialIterations, minimumRetryForNaN);
    }
}
