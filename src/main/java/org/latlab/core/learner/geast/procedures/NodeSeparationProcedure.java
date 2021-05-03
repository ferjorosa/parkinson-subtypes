package org.latlab.core.learner.geast.procedures;

import org.latlab.core.learner.geast.context.IProcedureContext;
import org.latlab.core.learner.geast.context.ISearchOperatorContext;
import org.latlab.core.learner.geast.operators.NodeSeparator;

public class NodeSeparationProcedure extends IterativeProcedure {

	public <T extends ISearchOperatorContext & IProcedureContext> NodeSeparationProcedure(
			T context) {
		super(context, new NodeSeparator(context));
	}

}
