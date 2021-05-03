/**
 * 
 */
package org.latlab.core.learner.geast.procedures;

import org.latlab.core.learner.geast.context.IProcedureContext;
import org.latlab.core.learner.geast.context.ISearchOperatorContext;

import java.util.Arrays;

/**
 * @author leonard
 * 
 */
public class SimplifyProcedure extends SequentialProcedure {

	/**
	 * @param context
	 */
	public <T extends ISearchOperatorContext & IProcedureContext> SimplifyProcedure(
			T context) {
		super(Arrays.asList(new NodeSeparationProcedure(context),
				new NodeDeletionProcedure(context), new StateDeletionProcedure(
						context)));
	}
}
