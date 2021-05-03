package org.latlab.core.learner.geast.procedures;

import org.latlab.core.learner.geast.context.IProcedureContext;
import org.latlab.core.learner.geast.context.ISearchOperatorContext;
import org.latlab.core.learner.geast.operators.NodeDeletor;

public class NodeDeletionProcedure extends IterativeProcedure {

	public <T extends ISearchOperatorContext & IProcedureContext> NodeDeletionProcedure(
			T context) {
		super(context, new NodeDeletor(context));
	}

}
