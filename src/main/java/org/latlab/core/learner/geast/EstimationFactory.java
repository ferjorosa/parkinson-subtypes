package org.latlab.core.learner.geast;

import org.latlab.core.data.MixedDataSet;
import org.latlab.core.learner.geast.SeparateTreePropagation.SharedData;
import org.latlab.core.model.Gltm;

/**
 * Creates {@code Estimation} instances for simple EM and restricted ML EM. It
 * uses {@link SharedTreePropagation} for simple EM and
 * {@link SeparateTreePropagation} for restricted ML EM.
 * 
 * 
 * @author leonard
 * 
 */
public class EstimationFactory {

	/**
	 * Prototype of the estimation. Use a different prototype to create a
	 * different type of {@code Estimation} instance.
	 * 
	 * @author leonard
	 * 
	 */
	public interface Prototype {
		Estimation create(Gltm model, Gltm clone, MixedDataSet data,
                          DataPropagation propagation, double smoothing);

		/**
		 * Number of partitions on the data for constructing the propagations.
		 * 
		 * @return number of partitions
		 */
		int partitions();
	}

	private final Prototype prototype;

	/**
	 * Accepts a prototype for determining which type of {@code Estimation} to
	 * create.
	 * 
	 * @param prototype
	 *            for creating a specific type of estimation.
	 */
	public EstimationFactory(Prototype prototype) {
		this.prototype = prototype;
	}

	/**
	 * Creates an estimation instance for simple EM.
	 * 
	 * @param model
	 *            original model for estimation
	 * @param data
	 *            data for estimation
	 * @param smoothing
	 *            the smoothing factor, currently unused
	 * @return
	 */
	public Estimation createSimple(Gltm model, MixedDataSet data,
			double smoothing) {
		Gltm clone = model.clone();
		DataPropagation propagation = new SharedTreePropagation(clone, data,
				prototype.partitions());
		return prototype.create(model, clone, data, propagation, smoothing);
	}

	/**
	 * Creates an estimation instance for restricted ML EM.
	 * 
	 * @param model
	 *            original model for estimation
	 * @param data
	 *            data for estimation
	 * @param smoothing
	 *            the smoothing factor, currently unused
	 * @return
	 */
	public Estimation createRestricted(Gltm model, MixedDataSet data,
			Focus focus, double smoothing) {
		return createRestricted(model, data, createSharedData(model, data,
				focus), smoothing);
	}

	public SharedData createSharedData(Gltm model, MixedDataSet data,
			Focus focus) {
		return SeparateTreePropagation.createSharedData(data, model, focus,
				prototype.partitions());
	}

	/**
	 * Creates an estimation instance for restricted ML EM, with an existing
	 * array of propagations.
	 * 
	 * @param model
	 *            original model for estimation
	 * @param data
	 *            data for estimation
	 * @param sharedData
	 *            used by the {@code SeparateTreePropagation}
	 * @param smoothing
	 *            the smoothing factor, currently unused
	 * @return
	 */
	public Estimation createRestricted(Gltm model, MixedDataSet data,
			SharedData sharedData, double smoothing) {
		Gltm clone = model.clone();
		DataPropagation propagation = new SeparateTreePropagation(clone, data,
				sharedData);
		return prototype.create(model, clone, data, propagation, smoothing);
	}
}
