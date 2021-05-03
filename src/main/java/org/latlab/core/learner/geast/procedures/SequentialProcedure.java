package org.latlab.core.learner.geast.procedures;

import org.latlab.core.learner.geast.IModelWithScore;
import org.latlab.core.learner.geast.operators.GivenCandidate;
import org.latlab.core.learner.geast.operators.SearchCandidate;

import java.util.List;

public class SequentialProcedure implements Procedure {

    private final List<? extends Procedure> procedures;
    private boolean succeeded = true;

    public SequentialProcedure(List<? extends Procedure> procedures) {
        this.procedures = procedures;
    }

    public void reset() {
        succeeded = true;
    }

    public SearchCandidate run(IModelWithScore base) {
        succeeded = false;
        SearchCandidate best = new GivenCandidate(base);
        
        for (Procedure procedure : procedures) {
            best = procedure.run(best.estimation());
            succeeded |= procedure.succeeded();
        }
        
        return best;
    }

    public boolean succeeded() {
        return succeeded;
    }

    public String name() {
        return getClass().getSimpleName();
    }

}
