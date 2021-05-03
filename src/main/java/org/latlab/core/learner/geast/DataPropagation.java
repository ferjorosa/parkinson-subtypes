package org.latlab.core.learner.geast;

import org.latlab.core.data.MixedDataSet;
import org.latlab.core.model.Gltm;
import org.latlab.core.reasoner.NaturalCliqueTree;
import org.latlab.core.reasoner.NaturalCliqueTreePropagation;

/**
 * Performs propagation on a data set. It provides a method for propagation on
 * each data case.
 * 
 * <p>
 * This class is used to abstract the combination of propagation and data set.
 * One implementation can share the same propagation for all data cases, while
 * another one can have each propagation for each data case.
 * 
 * @author leonard
 * 
 */
public abstract class DataPropagation {

	protected final Gltm model;
	protected MixedDataSet data;

	/**
	 * Constructs this object with a model and data set.
	 * 
	 * @param model
	 *            on which propagation is computed
	 * @param data
	 *            data for propagation
	 */
	public DataPropagation(Gltm model, MixedDataSet data) {
		this.model = model;
		this.data = data;
	}

	/**
	 * Returns a clique tree built using the {@link #model}. Only the structure
	 * of the clique tree should be used. Moreover, an estimation can contains
	 * more than one clique tree, so the node of the returned tree should not be
	 * used as any key in a map.
	 * 
	 * @return a clique tree built using the model
	 */
	public abstract NaturalCliqueTree cliqueTreeStructure();

	/**
	 * Returns the propagation computed on the given index of the data
	 * 
	 * @param index
	 * @return
	 */
	public abstract NaturalCliqueTreePropagation compute(int index);
}
