package org.latlab.core.learner.geast.procedures;

import org.latlab.core.learner.geast.context.IProcedureContext;
import org.latlab.core.learner.geast.context.ISearchOperatorContext;
import org.latlab.core.learner.geast.operators.StateIntroducer;

import java.util.Collections;

public class StateIntroductionProcedure extends IterativeProcedure {

	public <T extends ISearchOperatorContext & IProcedureContext> StateIntroductionProcedure(
			T context) {
		super(context, Collections.singletonList(new StateIntroducer(context)));
	}

}
