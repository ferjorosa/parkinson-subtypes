package eu.amidst.extension.util.mi.util;

import java.util.*;

public class DiscreteDataSet implements Iterable<double[]> {

    /** Index of each instance, from Hash to index */
    private Map<Double, Integer> index;

    private int currentMaxIndex;

    private List<double[]> instances;

    private List<Integer> counts;

    private int totalCounts;

    public DiscreteDataSet(double[][] data) {
        this.index = new HashMap<>();
        this.currentMaxIndex = 0;
        this.instances = new ArrayList<>();
        this.counts = new ArrayList<>();
        this.totalCounts = 0;

        for(int i=0; i < data.length; i++)
            add(data[i], 1);
    }

    public void add(double[] instance, int counts){

        this.totalCounts += counts;
        double instanceHash = Arrays.hashCode(instance);

        if(this.index.containsKey(instanceHash)){
            int instanceIndex = this.index.get(instanceHash);
            int instanceCount = this.counts.get(instanceIndex);
            this.counts.set(instanceIndex, instanceCount + counts);
        } else {
            this.instances.add(instance);
            this.counts.add(counts);
            this.index.put(instanceHash, currentMaxIndex);
            this.currentMaxIndex++;
        }
    }

    public int getCounts(double[] instance) {
        double instanceHash = Arrays.hashCode(instance);
        if(this.index.containsKey(instanceHash)){
            int instanceIndex = this.index.get(instanceHash);
            return this.counts.get(instanceIndex);
        }
        return 0;
    }

    public int getTotalCounts() {
        return totalCounts;
    }

    @Override
    public Iterator<double[]> iterator() {
        return instances.iterator();
    }
}
