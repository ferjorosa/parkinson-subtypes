package org.latlab.core.learner.geast;

import org.latlab.core.data.MixedDataSet;
import org.latlab.core.graph.AbstractNode;
import org.latlab.core.learner.geast.EstimationFactory.Prototype;
import org.latlab.core.model.Gltm;
import org.latlab.core.reasoner.CliqueTreeNode;
import org.latlab.core.reasoner.ImpossibleEvidenceException;
import org.latlab.core.reasoner.NaturalCliqueTreePropagation;
import org.latlab.core.reasoner.NaturalCliqueTreePropagation.ComputeTime;
import org.latlab.core.util.BuildConfig;

import java.util.Iterator;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;

public class MultithreadingEstimation extends Estimation {

    /**
     * Creates a factory that can be used to construct {@code
     * MultithreadingEstimation}.
     * 
     * @param threads
     *            number of threads used in estimation
     * @param executor
     *            usually given by {@code Executors.newFixedThreadPool(int)}
     * @return factory for constructing estimation instances
     */
    public static EstimationFactory createFactory(final int threads,
            final ExecutorService executor) {
        Prototype prototype = new Prototype() {
            public Estimation create(Gltm model, Gltm clone, MixedDataSet data,
                    DataPropagation propagation, double smoothing) {
                return new MultithreadingEstimation(model, clone, data,
                        propagation, threads, executor, smoothing);
            }

            public int partitions() {
                return threads;
            }
        };

        return new EstimationFactory(prototype);
    }

    /**
     * Holds the execution result after running a thread.
     * 
     * @author leonard
     * 
     */
    private static class Result {
        private RuntimeException exception = null;
    }

    private final ExecutorService executor;
    private final EqualPartitioner partitioner;
    private final int threads;

    private MultithreadingEstimation(Gltm origin, Gltm model, MixedDataSet data,
            DataPropagation propagation, int threads, ExecutorService executors,
            double smoothing) {
        super(origin, model, data, propagation, smoothing);

        this.threads = threads;
        this.executor = executors;
        this.partitioner = new EqualPartitioner(data.size(), threads);
    }

    @Override
    protected void computeSufficientStatistics() {
        messagesPassed = 0;

        ExecutorCompletionService<Result> ecs =
                new ExecutorCompletionService<Result>(executor);

        for (int i = 0; i < threads; i++) {
            Result result = new Result();
            ecs.submit(createRunnable(i, result), result);
        }

        Result[] results = new Result[threads];

        for (int i = 0; i < threads; i++) {
            try {
                results[i] = ecs.take().get();
            } catch (InterruptedException e) {
                e.printStackTrace(Log.errorWriter());
            } catch (ExecutionException e) {
                e.printStackTrace(Log.errorWriter());
            }
        }

        checkException(results);
    }

    /**
     * Checks whether the results have any exception, and throws it if they
     * have.
     * 
     * <p>
     * Note: In case multiple exceptions are found, it only throws the first
     * exception that is not {@link ImpossibleEvidenceException} if there exists
     * one, or the first {@code ImpossibleEvidenceException} if there does not.
     * 
     * @param results
     *            execution results of {@link #createRunnable(int)}
     */
    private void checkException(Result[] results) {
        for (Result result : results) {
            if (!(result.exception == null
                    || result.exception instanceof ImpossibleEvidenceException))
                throw new RuntimeException(result.exception);
        }

        for (Result result : results) {
            if (result.exception != null)
                throw result.exception;
        }
    }

    private Runnable createRunnable(final int subset, final Result result) {
        return new Runnable() {
            public void run() {
                try {
                    computeSufficientStatistics(subset);
                } catch (RuntimeException e) {
                    result.exception = e;
                }
            }
        };
    }

    /**
     * Computes sufficient statistics on the subset of data with the given
     * index.
     * 
     * @param subset
     *            index of the subset of data
     */
    private void computeSufficientStatistics(int subset) {
        int start = partitioner.startOf(subset);
        int end = partitioner.endOf(subset);

        ComputeTime time = BuildConfig.PROFILE ? new ComputeTime() : null;

        for (int i = start; i < end; i++) {
            NaturalCliqueTreePropagation ctp = propagation.compute(i);

            if (BuildConfig.PROFILE) {
                time.add(ctp.computeTime);
            }

            addMessageCount(ctp.messagesPassed());

            Iterator<SufficientStatistics> statisticsIterator =
                    sufficientStatistics.iterator();

            for (AbstractNode node : ctp.cliqueTree().getNodes()) {
                final SufficientStatistics statistics =
                        statisticsIterator.next();

                // make sure that the same statistics instance is not accessed
                // by more than one threads simultaneously
                synchronized (statistics) {
                    statistics.add(((CliqueTreeNode) node).potential(),
                            data.get(i).weight());
                }
            }

            double weight = data.get(i).weight();

            synchronized (this) {
                loglikelihood += ctp.loglikelihood() * weight;
            }
        }

        BuildConfig.profileLog(ComputeTime.format(time));
    }

    private synchronized void addMessageCount(int messages) {
        messagesPassed += messages;
    }
}
