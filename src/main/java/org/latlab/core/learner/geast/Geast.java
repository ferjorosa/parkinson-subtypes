/**
 * 
 */
package org.latlab.core.learner.geast;

import org.latlab.core.data.MixedDataSet;
import org.latlab.core.learner.geast.context.Context;
import org.latlab.core.learner.geast.procedures.AdjustProcedure;
import org.latlab.core.learner.geast.procedures.ExpandProcedure;
import org.latlab.core.learner.geast.procedures.Procedure;
import org.latlab.core.learner.geast.procedures.SimplifyProcedure;
import org.latlab.core.model.Gltm;

import java.io.PrintWriter;

/**
 * Implementation of the GEAST algorithm.
 * 
 * <p>
 * It contains some {@link Procedure} for improving the models. Each procedure
 * is run consecutively (with cycle) to improve the current model. This
 * algorithm stops when all procedures fail to find any better model.
 * 
 * @author leonard
 * 
 */
public class Geast {

	public static final int DEFAULT_THREADS = 1;
	public static final int DEFAULT_SCREENING = 64;
	public static final double DEFAULT_THRESHOLD = 1e-2;

	public String commandLine;

	private final static String ELEMENT = "geast";

	private final Procedure[] procedures;

	private final Context context;

	public Geast(Context context) {
		this(context, new Procedure[] { new ExpandProcedure(context),
				new AdjustProcedure(context), new SimplifyProcedure(context) });
	}

	public Geast(Context context, Procedure[] procedures) {
		this.context = context;
		this.procedures = procedures;
	}

	public Geast(int threads, int screening, double threshold,
			MixedDataSet data, Log log, EmFramework screeningEm,
			EmFramework selectionEm, EmFramework estimationEm) {
		this(new Context(threads, screening, threshold, data, log, screeningEm,
				selectionEm, estimationEm));
	}

	public Geast(int threads, int screening, double threshold,
				 MixedDataSet data, Log log, EmFramework screeningEm,
				 EmFramework selectionEm, EmFramework estimationEm, long seed) {
		this(new Context(threads, screening, threshold, data, log, screeningEm,
				selectionEm, estimationEm, seed));
	}

	public Geast(int threads, MixedDataSet data, Log log) {
		this(threads, DEFAULT_SCREENING, DEFAULT_THRESHOLD, data, log,
				new LocalEm(data, false, 1, 40, 0.01), new LocalEm(data, true,
						32, 50, 0.01), new FullEm(data, true, 64, 500, 0.01));
	}

	/**
	 * For testing purpose. It allows specifying some mock procedures so that
	 * the algorithm can complete quickly.
	 */
	Geast(MixedDataSet data, Log log, EmFramework screeningEm,
			EmFramework selectionEm, EmFramework estimationEm,
			Procedure[] procedures) {
		this(new Context(DEFAULT_THREADS, 64, 0.01, data, log, screeningEm,
				selectionEm, estimationEm), procedures);
	}

	/**
	 * Learns and returns a model using this algorithm.
	 * 
	 * @return model learned from this algorithm
	 */
	public IModelWithScore learn() {
		Gltm initial =
				Gltm.constructLocalIndependenceModel(context.data().getNonClassVariables());
		context.parameterGenerator().generate(initial);
		return learn(initial);
	}

	/**
	 * Learns and returns a model using this algorithm starting from the given
	 * initial model.
	 * 
	 * @param initial
	 *            from which this algorithm starts
	 * @return model learned by this algorithm
	 */
	public IModelWithScore learn(Gltm initial) {
		if (context.executor().isShutdown()) {
			throw new UnsupportedOperationException(
					"A Geast instance cannot be used to learn a model "
							+ "for more than one times.");
		}

		context.log().start();
		context.log().writeStartElementWithTime(
				ELEMENT,
				String.format("startTime='%s'",
						Log.LOG_DATE_FORMAT.format(context.log().startTime())));
		context.log().writeCommandLine(commandLine);
		context.log().writeDataElement(context.data());
		writeSettingsXml(context.log().writer());

		IModelWithScore current = null;

		try {
			current = context.estimationEm().estimate(initial);
			context.log().writeElementWithEstimationToFile("initial", current,
					"initial", false);

			boolean stop = false;

			// run the procedures initially. This sets the succeeded flags of
			// the procedures and determine next whether a procedure should
			// continue to run
			for (Procedure procedure : procedures)
				current = procedure.run(current).estimation();

			while (!stop) {
				// continue to run the procedures until the procedures other
				// than the current one fails to find a better model
				for (Procedure procedure : procedures) {
					if (shouldStop(procedure)) {
						stop = true;
						break;
					}

					current = procedure.run(current).estimation();
				}
			}
			
			// this version does not run FULL EM on the candidates.  So here we
			// run it.
			
			current = context.estimationEm().estimate(current);

			return current;

		} catch (EstimationNaNException e) {
			context.log().write(e, e.estimation(), null);
			throw e;

		} catch (RuntimeException e) {
			context.log().write(e);
			throw e;

		} finally {
			context.executor().shutdown();

			if (current != null)
				context.log().writeElementWithEstimationToFile("final",
						current, "final", false);

			context.log().writeEndElement(ELEMENT);
			context.log().close();
		}

	}

	/**
	 * Checks whether it should stop the search. It stops for a procedure when
	 * others have failed to find a better model, so that the given {@code
	 * procedure} doesn't have to search again.
	 * 
	 * @param procedure
	 *            procedure that is preparing to start
	 * @return whether it should stop the search
	 */
	private boolean shouldStop(Procedure procedure) {
		for (Procedure p : procedures) {
			if (p != procedure && p.succeeded())
				return false;
		}

		return true;
	}

	/**
	 * This method is provided for testing basically.
	 * 
	 * @return the context of this GEAST algorithm.
	 */
	public Context context() {
		return context;
	}

	/**
	 * Writes the setting in XML format.
	 * 
	 * @param writer
	 *            for writing to output
	 */
	private void writeSettingsXml(PrintWriter writer) {
		writer.format(
				"<settings threads='%d' screening='%d' threshold='%.2e'>",
				context.threads, context.screeningSize(), context.threshold());
		writer.println();
		context.covarianceConstrainer().writeXml(writer);
		context.screeningEm().writeXml(writer, "screening");
		context.selectionEm().writeXml(writer, "selection");
		context.estimationEm().writeXml(writer, "estimation");
		writer.println("</settings>");
	}

}
