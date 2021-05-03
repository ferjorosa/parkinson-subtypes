package org.latlab.core.learner.geast;

import org.latlab.core.data.MixedDataSet;
import org.latlab.core.learner.geast.context.Context;
import org.latlab.core.learner.geast.operators.NodeCombiner;
import org.latlab.core.learner.geast.operators.SearchOperator;
import org.latlab.core.learner.geast.procedures.IterativeProcedure;
import org.latlab.core.learner.geast.procedures.Procedure;
import org.latlab.core.model.Builder;
import org.latlab.core.model.Gltm;

import java.util.Arrays;


// not finished!

public class MultimixLearner {
    private final Geast geast;
    private final int components;

    /**
     * 
     * @param data
     * @param log
     * @param components
     * @param increase
     *            whether to increase the number of states
     */
    public MultimixLearner(MixedDataSet data, Log log, int components) {
        this(
            Geast.DEFAULT_THREADS, Geast.DEFAULT_SCREENING,
            Geast.DEFAULT_THRESHOLD, data, log, new FullEm(
                data, true, 64, 500, 0.01), components);
    }

    public MultimixLearner(
        int threads, int screening, double threshold, MixedDataSet data,
        Log log, EmFramework em, int components) {
        Context context =
            new Context(threads, screening, threshold, data, log, em, em, em);
        SearchOperator operator = new NodeCombiner(context);
        Procedure procedure =
            new IterativeProcedure(context, Arrays.asList(operator));
        Procedure[] procedures = new Procedure[] { procedure };

        geast = new Geast(context, procedures);

        this.components = components;
    }

    public IModelWithScore learn() {
        Gltm initial =
            Builder.buildMixedMixtureModel(
                new Gltm(), components, geast.context().data()
                    .getNonClassVariables());
        return geast.learn(initial);
    }

    public void setCommandLine(String commandLine) {
        geast.commandLine = commandLine;
    }
}
