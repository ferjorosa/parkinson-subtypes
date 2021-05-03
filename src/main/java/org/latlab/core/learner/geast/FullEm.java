package org.latlab.core.learner.geast;

import org.latlab.core.data.MixedDataSet;
import org.latlab.core.model.Gltm;

public class FullEm extends EmFramework {

    public FullEm(MixedDataSet data, boolean reuseParameters, int restarts, int maxSteps, double threshold) {
        super(data);
        use(new EmParameters(reuseParameters, restarts, 0, maxSteps, threshold));
    }

    public FullEm(MixedDataSet data) {
        this(data, true, 64, 500, 1e-2);
    }

    /**
     * Focus is not used in full EM.
     */
    @Override
    protected Estimation[] createEstimations(int size, Gltm model, Focus focus) {
        Estimation[] estimations = new Estimation[size];
        for (int i = 0; i < estimations.length; i++)
            estimations[i] =
                estimationFactory().createSimple(
                    model, data, parameters.smoothing);

        return estimations;
    }
}
