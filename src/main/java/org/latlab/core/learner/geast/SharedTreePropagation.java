package org.latlab.core.learner.geast;

import org.latlab.core.data.MixedDataSet;
import org.latlab.core.model.Gltm;
import org.latlab.core.reasoner.NaturalCliqueTree;
import org.latlab.core.reasoner.NaturalCliqueTreePropagation;

/**
 * Uses a single (or a fixed number of) clique tree (setting to different
 * evidences for each time) to compute all the data cases.
 * 
 * <p>
 * If more than one clique trees are used, the data cases are partitioned into
 * equal segments and each clique tree is used for each segment.  This is 
 * necessary to do inference on the data cases in parallel by multiple threads.
 * 
 * @author leonard
 * 
 */
public class SharedTreePropagation extends DataPropagation {

    private final NaturalCliqueTreePropagation[] propagations;
    private final EqualPartitioner partitioner;

    public SharedTreePropagation(Gltm model, MixedDataSet data) {
        this(model, data, 1);
    }

    /**
     * Constructs this shared tree propagation with given model, data, and
     * number of partitions in the data.
     * 
     * @param model
     *            on which messages propagate on
     * @param data
     *            on which propagation is computed
     * @param number
     *            number of partitions in data. Each partition must use a
     *            separate clique tree.
     */
    public SharedTreePropagation(Gltm model, MixedDataSet data, int number) {
        super(model, data);

        propagations = new NaturalCliqueTreePropagation[number];

        for (int i = 0; i < propagations.length; i++)
            propagations[i] = new NaturalCliqueTreePropagation(this.model);

        partitioner = new EqualPartitioner(data.size(), number);
    }

    @Override
    public NaturalCliqueTree cliqueTreeStructure() {
        return propagations[0].cliqueTree();
    }

    @Override
    public NaturalCliqueTreePropagation compute(int index) {
        NaturalCliqueTreePropagation propagation = propagations[partitioner.indexOf(index)];
        propagation.use(data.getEvidences(index));
        propagation.propagate();
        return propagation;
    }
}
