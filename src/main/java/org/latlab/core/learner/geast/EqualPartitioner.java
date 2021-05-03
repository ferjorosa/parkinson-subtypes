package org.latlab.core.learner.geast;

/**
 * Partitions a set of objects into a given number of subsets of equal size. If
 * the set cannot be divided into equal size, all subsets have the equal size
 * except the last one.
 * 
 * <p>
 * This class makes other classes using a partitioner with the same settings
 * have the same partitioning.
 * 
 * @author leonard
 * 
 */
public class EqualPartitioner {
	private final int sizeOfSubset;
	private final int size;

	public EqualPartitioner(int size, int subsets) {
		this.size = size;
		sizeOfSubset = (int) Math.ceil((double) size / subsets);
	}

	/**
	 * This class gives the index of the subset an item belongs to given the
	 * index of the item in the original set.
	 */
	public int indexOf(int originalIndex) {
		return originalIndex / sizeOfSubset;
	}

	public int startOf(int subset) {
		return sizeOfSubset * subset;
	}

	public int endOf(int subset) {
		return Math.min(sizeOfSubset * (subset + 1), size);
	}
}
