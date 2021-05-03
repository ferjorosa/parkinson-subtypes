package eu.amidst.extension.missing.util;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.distribution.Multinomial;
import eu.amidst.core.distribution.Normal;
import eu.amidst.core.distribution.UnivariateDistribution;
import eu.amidst.core.inference.messagepassing.VMP;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.variables.HashMapAssignment;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.data.DataUtils;
import eu.amidst.extension.util.MyMath;

import java.util.*;
import java.util.stream.Collectors;

/**
 *
 */
public class ImputeMissing {

    /**
     * Learns a univariate distribution for each attribute and completes missing values with a sample from it.
     */
    public static DataOnMemory<DataInstance> imputeWithMarginals(DataOnMemory<DataInstance> data, int seed) {

        DataOnMemory<DataInstance> copyData = data.deepCopy();

        /* Iterate through the set of attributes to estimate their marginal distributions */
        Variables variables = new Variables();
        List<UnivariateDistribution> dists = new ArrayList<>(copyData.getAttributes().getNumberOfAttributes());
        for(Attribute attribute: copyData.getAttributes()) {
            UnivariateDistribution dist = estimateUnivariateDistribution(copyData, attribute, variables);
            dists.add(dist);
        }

        /* Substitute missing values with a sample from its corresponding marginal distribution */
        Random random = new Random(seed);
        for(DataInstance instance: copyData) {
            double[] values = instance.toArray();
            for(int i=0; i <values.length;i++) {
                if (Double.isNaN(values[i])) {
                    values[i] = dists.get(i).sample(random);
                }
            }
        }

        return copyData;
    }



    public static DataOnMemory<DataInstance> imputeWithModel(DataOnMemory<DataInstance> dataWithMissing,
                                                             LinkedHashMap<Integer, List<Attribute>> instancesWithMissing,
                                                             BayesianNetwork model){

        DataOnMemory<DataInstance> copyData = dataWithMissing.deepCopy();

        /* Impute missing data using inference */
        VMP vmp = new VMP();
        vmp.setModel(model);
        for(int instIndex: instancesWithMissing.keySet()) {

            List<Variable> missingVariables = instancesWithMissing.get(instIndex).stream()
                    .map(attribute -> model.getVariables().getVariableByName(attribute.getName()))
                    .collect(Collectors.toList());

            /* Prepare instance for inference. First we add all values and then remove missing ones */
            DataInstance instance = copyData.getDataInstance(instIndex);
            HashMapAssignment assignment = new HashMapAssignment(copyData.getAttributes().getNumberOfAttributes() - missingVariables.size());
            for(Variable var: model.getVariables())
                assignment.setValue(var, instance.getValue(var));
            for(Variable var: missingVariables)
                assignment.removeValue(var);

            /*
             * Hard imputation using inference:
             *   - Obtain Posterior distribution
             *   - Assign expected value (highest prob index for discrete, mean for continuous)
-            */
            vmp.setEvidence(assignment);
            vmp.runInference();
            for(Variable var: missingVariables) {
                UnivariateDistribution posterior = vmp.getPosterior(var);
                instance.setValue(var, expectedValue(posterior));
            }
        }

        return copyData;
    }

    private static UnivariateDistribution estimateUnivariateDistribution(DataOnMemory<DataInstance> data,
                                                                         Attribute attribute,
                                                                         Variables variables) {
        /* Project the dataset to contain the selected attribute */
        List<Attribute> attributesForProjection = new ArrayList<>(1);
        attributesForProjection.add(attribute);
        DataOnMemory<DataInstance> projectedData = DataUtils.project(data, attributesForProjection);
        Attribute projectedAttribute = projectedData.getAttributes().getAttributeByName(attribute.getName());

        /* Remove instances with missing values (only one variable) */
        DataOnMemory<DataInstance> noMissing = projectedData.filter(x-> !Double.isNaN(x.getValue(projectedAttribute))).toDataOnMemory();

        /* If the attribute is continuous, estimate a Gaussian distribution */
        if(projectedAttribute.isContinuous()) {

            /* Using MyMath, estimate its mean and variance */
            double[] values = new double[noMissing.getNumberOfDataInstances()];
            for(int i=0; i < values.length; i++)
                values[i] = noMissing.getDataInstance(i).getValue(projectedAttribute);
            double mean = MyMath.mean(values);
            double stdev = MyMath.stDev(values, mean);
            double variance = stdev * stdev;

            /* Parametrize a Gaussian distribution */
            Variable variable = variables.newGaussianVariable(projectedAttribute);
            Normal gaussian = new Normal(variable);
            gaussian.setMean(mean);
            gaussian.setVariance(variance);

            return gaussian;
        }
        /* If the attribute is discrete, estimate a Multinomial distribution */
        else {

            /* Initialize counts with a Laplace smooth */
            Map<Integer, Integer> counts = new HashMap<>();
            for(int i = 0; i < projectedAttribute.getNumberOfStates();i++)
                counts.put(i, 1);

            /* Estimate marginal counts */
            for(DataInstance instance: noMissing){
                int value = (int) instance.getValue(projectedAttribute);
                int count = counts.get(value) + 1;
                counts.put(value, count);
            }

            /* Estimate marginal frequencies */
            double[] frequencies = new double[counts.keySet().size()];
            int n = 0;
            for(int i: counts.keySet())
                n += counts.get(i);
            for(int i: counts.keySet())
                frequencies[i] = (double) counts.get(i) / n;

            /* Parametrize a Multinomial distribution */
            Variable variable = variables.newMultinomialVariable(projectedAttribute);
            Multinomial multinomial = new Multinomial(variable);
            multinomial.setProbabilities(frequencies);

            return multinomial;
        }
    }

    /**
     * Complete missing values the following way:
     *      - Continuous: mean value of the attribute.
     *      - Discrete: most common state of the attribute.
     */
    public static DataOnMemory<DataInstance> imputeWithExpectedValue(DataOnMemory<DataInstance> data) {

        DataOnMemory<DataInstance> copyData = data.deepCopy();

        /* Iterate through the set of attributes to estimate their expected values */
        List<Double> expectedValues = new ArrayList<>(copyData.getAttributes().getNumberOfAttributes());
        for(Attribute attribute: copyData.getAttributes())
            expectedValues.add(estimateExpectedValue(copyData, attribute));

        /* Substitute missing values with its attribute's expected value */
        for(DataInstance instance: copyData) {
            double[] values = instance.toArray();
            for(int i=0; i < values.length; i++) {
                if (Double.isNaN(values[i])) {
                    values[i] = expectedValues.get(i);
                }
            }
        }

        return copyData;
    }

    private static double estimateExpectedValue(DataOnMemory<DataInstance> data,
                                                Attribute attribute) {

        /* Project the dataset to contain the selected attribute */
        List<Attribute> attributesForProjection = new ArrayList<>(1);
        attributesForProjection.add(attribute);
        DataOnMemory<DataInstance> projectedData = DataUtils.project(data, attributesForProjection);

        /* Remove instances with missing values */
        DataOnMemory<DataInstance> noMissing = projectedData.filter(x-> !Double.isNaN(x.toArray()[0])).toDataOnMemory();

        /* If the attribute is continuous, estimate its mean */
        if(attribute.isContinuous()) {
            double[] values = new double[noMissing.getNumberOfDataInstances()];
            for(int i=0; i < values.length; i++)
                values[i] = noMissing.getDataInstance(i).toArray()[0];
            return MyMath.mean(values);
        }
        /* If the attribute is discrete, estimate its most frequent state */
        else {

            Map<Integer, Integer> counts = new HashMap<>();
            for(int i = 0; i < attribute.getNumberOfStates();i++)
                counts.put(i, 0);

            for(DataInstance instance: noMissing){
                int value = (int) instance.toArray()[0];
                int count = counts.get(value) + 1;
                counts.put(value, count);
            }

            int currentIndex = 0;
            int currentCounts = 0;
            for(int index: counts.keySet()){
                if(counts.get(index) > currentCounts) {
                    currentCounts = counts.get(index);
                    currentIndex = index;
                }
            }

            return currentIndex;
        }
    }

    public static DataOnMemory<DataInstance> imputeWithModel(DataOnMemory<DataInstance> dataWithMissing,
                                                              BayesianNetwork model) {

        DataOnMemory<DataInstance> imputedData = dataWithMissing.deepCopy();

        /* Impute missing data using inference */
        VMP vmp = new VMP();
        vmp.setModel(model);

        for(int i = 0; i<dataWithMissing.getNumberOfDataInstances(); i++) {

            List<Variable> missingVariables = new ArrayList<>();
            HashMapAssignment evidence = new HashMapAssignment();

            /* Find missing values for inference */
            Variable var = null;
            double value = Double.NaN;
            for(int j = 0; j < imputedData.getAttributes().getNumberOfAttributes(); j++) {
                var = model.getVariables().getListOfVariables().get(j);
                value = dataWithMissing.getDataInstance(i).toArray()[j];
                if(Double.isNaN(value))
                    missingVariables.add(var);
                else
                    evidence.setValue(var, value);
            }

            /*
             * Hard imputation using inference:
             *   - Obtain Posterior distribution
             *   - Assign expected value (highest prob index for discrete, mean for continuous)
-            */
            vmp.setEvidence(evidence);
            vmp.runInference();
            for(Variable missingVar: missingVariables) {
                UnivariateDistribution posterior = vmp.getPosterior(missingVar);
                imputedData.getDataInstance(i).setValue(missingVar, expectedValue(posterior));
            }
        }

        return imputedData;
    }

    /** Return the expected value of an univariate distribution */
    private static double expectedValue(UnivariateDistribution distribution) {
        /* If it is discrete, we have a multinomial dist, thus we return its highest probability index */
        if(distribution.getVariable().isDiscrete()) {
            int highestProbIndex = 0;
            double highestProb = 0.0;
            double[] parameters = distribution.getParameters();
            for(int i=0; i < parameters.length; i++)
                if(parameters[i] > highestProb) {
                    highestProb = parameters[i];
                    highestProbIndex = i;
                }
            return highestProbIndex;

            /* If it is continuous, we have a Gaussian dist, thus we return its mean, which is the firs parameter */
        } else
            return distribution.getParameters()[0];
    }
}
