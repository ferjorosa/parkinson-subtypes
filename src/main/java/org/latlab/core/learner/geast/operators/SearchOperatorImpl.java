/**
 * 
 */
package org.latlab.core.learner.geast.operators;

import org.latlab.core.learner.geast.context.ISearchOperatorContext;

/**
 * Implementation of the search operator. 
 * 
 * <p>
 * All subclasses of search operator
 * should derive from this class so that they have the same kind of
 * implementation. This class derives from either
 * {@link SinglethreadSearchOperator} or {@link MultithreadSearchOperator} to
 * determine whether to search using single thread or multiple threads.
 * 
 * @author leonard
 * 
 */
public abstract class SearchOperatorImpl extends SinglethreadSearchOperator {

    public SearchOperatorImpl(ISearchOperatorContext context) {
        super(context);
    }

}
