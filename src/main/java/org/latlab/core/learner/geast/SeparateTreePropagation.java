package org.latlab.core.learner.geast;

import org.latlab.core.data.MixedDataSet;
import org.latlab.core.model.Gltm;
import org.latlab.core.reasoner.NaturalCliqueTree;
import org.latlab.core.reasoner.NaturalCliqueTreePropagation;
import org.latlab.core.reasoner.NaturalCliqueTreePropagation.LocalPropagationMemento;

/**
 * It uses a separate tree for each data case, so that some previous computation
 * with the same evidence can be stored in the tree.
 * 
 * <p>
 * The set of propagation trees with the same focus can be shared by different
 * models with different parameters inside the focus. It sets the model of the
 * propagation trees use to the one referenced by this propagation when it
 * computes for one data case.
 * 
 * <p>
 * The models sharing the propagation trees are assumed to have the same overall
 * structure and parameters outside the focus. Due to this, the evidences can be
 * absorbed only once for each data case even with different models.
 * 
 * TODO LP: uses only one clique tree for all data cases, but stores the
 * potentials in the separators for each data case.
 * 
 * @author leonard
 * 
 */
public class SeparateTreePropagation extends DataPropagation {
	private final SharedData sharedData;

	public static class SharedData {
		private final LocalPropagationMemento[] mementos;
		private final NaturalCliqueTreePropagation[] propagations;
		private final EqualPartitioner partitioner;

		/**
		 * 
		 * @param data
		 * @param model
		 * @param focus
		 * @param number
		 *            number of partitions, each of which uses one clique tree
		 */
		private SharedData(MixedDataSet data, Gltm model, Focus focus,
				int number) {
			mementos = new LocalPropagationMemento[data.size()];
			propagations = new NaturalCliqueTreePropagation[number];
			for (int i = 0; i < number; i++) {
				propagations[i] = new NaturalCliqueTreePropagation(model, focus);
			}

			partitioner = new EqualPartitioner(data.size(), number);
		}
	}

	public static SharedData createSharedData(MixedDataSet data, Gltm model,
			Focus focus, int number) {
		return new SharedData(data, model, focus, number);
	}

	public SeparateTreePropagation(Gltm model, MixedDataSet data,
			SharedData shared) {
		super(model, data);

		if (data.size() == 0) {
			throw new IllegalArgumentException(
					"The data contains no data case.");
		}

		this.sharedData = shared;
	}

	public SeparateTreePropagation(Gltm model, MixedDataSet data, Focus focus,
			int number) {
		this(model, data, new SharedData(data, model, focus, number));
	}

	@Override
	public NaturalCliqueTree cliqueTreeStructure() {
		return sharedData.propagations[0].cliqueTree();
	}

	/**
	 * Note: this method is accessed by multiple threads.
	 */
	@Override
	public NaturalCliqueTreePropagation compute(int index) {
		// use different propagation instance for different thread
		int partition = sharedData.partitioner.indexOf(index);
		NaturalCliqueTreePropagation propagation = sharedData.propagations[partition];

		// set the model and evidence since this propagation object can be
		// shared by different models and evidences
		propagation.useModel(model);
		propagation.use(data.getEvidences(index));

		if (sharedData.mementos[index] == null) {
			// this is the first time of propagation on this evidence, stores
			// the memento for later use

			propagation.resetLocalPropagation();
			propagation.propagate();

			propagation.releaseSeparatorMessagesOutsideFocus();
			sharedData.mementos[index] = propagation
					.createLocalPropagationMemento();
		} else {
			// this is not the first time, recover from the previous local
			// propagation
			propagation.setLocalPropagationMemento(sharedData.mementos[index]);
			propagation.propagate();
		}

		return propagation;
	}

}
