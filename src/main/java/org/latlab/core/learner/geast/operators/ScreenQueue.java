package org.latlab.core.learner.geast.operators;

import org.latlab.core.util.BoundedQueue;

/**
 * A queue holding the search candidates and their estimations for screening.
 * 
 * @author leonard
 * 
 */
public class ScreenQueue extends BoundedQueue<SearchCandidate> {

    public ScreenQueue(int maximumSize) {
        super(maximumSize, SearchCandidate.SCORE_COMPARATOR);
    }
//
//    /**
//     * Adds a candidate model to this queue and uses the EM algorithm to
//     * estimate the parameters and its score.
//     * 
//     * @param candidate
//     *            candidate to add to the queue.
//     */
//    public boolean add(Candidate candidate) {
//        candidate.estimate(em);
//        super.add(candidate);
//        return true;
//    }

}
