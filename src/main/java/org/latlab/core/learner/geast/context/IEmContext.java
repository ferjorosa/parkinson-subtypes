package org.latlab.core.learner.geast.context;

import org.latlab.core.learner.geast.EmFramework;

import java.util.concurrent.Executor;

public interface IEmContext {

	public EmFramework screeningEm();

	public EmFramework selectionEm();

	public Executor searchExecutor();

}
