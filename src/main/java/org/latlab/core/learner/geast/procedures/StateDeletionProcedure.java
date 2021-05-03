package org.latlab.core.learner.geast.procedures;

import org.latlab.core.learner.geast.context.IProcedureContext;
import org.latlab.core.learner.geast.context.ISearchOperatorContext;
import org.latlab.core.learner.geast.operators.StateDeletor;

public class StateDeletionProcedure extends IterativeProcedure {
	public <T extends ISearchOperatorContext & IProcedureContext> StateDeletionProcedure(
			T context) {
		super(context, new StateDeletor(context));
	}
}
