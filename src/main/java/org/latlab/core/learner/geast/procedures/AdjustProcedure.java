/**
 * 
 */
package org.latlab.core.learner.geast.procedures;

import org.latlab.core.learner.geast.context.IProcedureContext;
import org.latlab.core.learner.geast.context.ISearchOperatorContext;
import org.latlab.core.learner.geast.operators.NodeRelocator;

/**
 * @author leonard
 * 
 */
public class AdjustProcedure extends IterativeProcedure {

	/**
	 * @param context
	 */
	public <T extends IProcedureContext & ISearchOperatorContext> AdjustProcedure(
			T context) {
		super(context, new NodeRelocator(context));
	}
}
