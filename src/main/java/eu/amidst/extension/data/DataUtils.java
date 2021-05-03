package eu.amidst.extension.data;

import eu.amidst.core.datastream.*;
import eu.amidst.core.distribution.Multinomial;
import eu.amidst.core.distribution.Normal;
import eu.amidst.core.inference.InferenceEngine;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.stateSpaceTypes.RealStateSpace;
import eu.amidst.extension.util.tuple.Tuple2;

import java.util.ArrayList;
import java.util.List;

public class DataUtils {

    /**
     * Returns a new dataset with data of only the selected attributes.
     */
    public static DataOnMemory<DataInstance> project(DataOnMemory<DataInstance> data, List<Attribute> attributesForProjection) {

        if(!data.getAttributes().getFullListOfAttributes().containsAll(attributesForProjection))
            throw new IllegalArgumentException("Not all the attributes are present in the data");

        /* Clonamos los atributos pero les asignamos un nuevo indice*/
        List<Attribute> projectedAttributesList = new ArrayList<>();
        for(int i = 0; i < attributesForProjection.size(); i++){
            Attribute attribute = attributesForProjection.get(i);
            projectedAttributesList.add(new Attribute(i, attribute.getName(), attribute.getStateSpaceType()));
        }

        Attributes projectedAttributes = new Attributes(projectedAttributesList);
        List<DataInstance> dataInstanceList = new ArrayList<>();

        for(DataInstance dataInstance: data) {
            double[] projectedData = new double[projectedAttributes.getNumberOfAttributes()];
            for (int i = 0; i < projectedAttributes.getNumberOfAttributes(); i++) {
                projectedData[i] = dataInstance.getValue(attributesForProjection.get(i));
            }
            dataInstanceList.add(new DataInstanceFromRawData(projectedAttributes, projectedData));
        }

        return new DataOnMemoryListContainer<>(projectedAttributes, dataInstanceList);
    }

    /**
     * Returns a new dataset with the original attributes plus a new one for each latent variable. Data from latent
     * variables have been completed using their MAP estimates.
     */
    public static DataOnMemory<DataInstance> completeLatentData(DataOnMemory<DataInstance> data, BayesianNetwork latentModel) {

        /* Select latent variables in the model and create a new attribute for each of them. Add all attributes
         * (new and old) to a new Attribute list */
        List<Attribute> newAttributeList = new ArrayList<>();
        List<Variable> latentVariables = new ArrayList<>();
        int currentIndex = data.getAttributes().getNumberOfAttributes();
        for(Variable variable: latentModel.getVariables()) {
            if (!variable.isObservable()) {
                latentVariables.add(variable);
                Attribute newAttribute = new Attribute(currentIndex, variable.getName(), variable.getStateSpaceType());
                newAttributeList.add(newAttribute);
                currentIndex++;
            } else
                newAttributeList.add(variable.getAttribute());
        }

        Attributes newAttributes = new Attributes(newAttributeList);
        DataOnMemoryListContainer<DataInstance> completeData = new DataOnMemoryListContainer<>(newAttributes);

        /* Iterate through the set of data instances and complete data using the MAP estimate */
        for(DataInstance instance: data) {
            double[] newInstanceValues = new double[newAttributes.getNumberOfAttributes()];

            /* First, add all the observed data */
            int originalNumberOfAttributes = data.getAttributes().getNumberOfAttributes();
            for(int i = 0; i < originalNumberOfAttributes; i++)
                newInstanceValues[i] = instance.getValue(data.getAttributes().getFullListOfAttributes().get(i));

            /* Second, estimate the MAP values of latent variables and add them to the instance */
            for(int i = originalNumberOfAttributes; i < (originalNumberOfAttributes + latentVariables.size()); i++){
                Variable latentVariable = latentModel.getVariables().getListOfVariables().get(i);
                if(latentVariable.isDiscrete()) {
                    Multinomial posterior = InferenceEngine.getPosterior(latentVariable, latentModel, instance);
                    int indexMaxProbability = maxProbIndex(posterior.getProbabilities()); // Hard assignment
                    newInstanceValues[i] = indexMaxProbability;
                } else {
                    Normal posterior = InferenceEngine.getPosterior(latentVariable, latentModel, instance);
                    newInstanceValues[i] = posterior.getMean();
                }
            }

            completeData.add(new DataInstanceFromRawData(newAttributes, newInstanceValues));
        }

        return completeData;
    }

    /**
     * Returns a new dataset with ONLY the completed data from the model's latent variables.
     */
    public static DataOnMemory<DataInstance> completeDiscreteLatent(DataOnMemory<DataInstance> data, BayesianNetwork latentModel) {
        List<Variable> discreteLatentVariables = new ArrayList<>();
        for(Variable variable: latentModel.getVariables())
            if(variable.isDiscrete() && !variable.isObservable())
                discreteLatentVariables.add(variable);

        return completeDiscreteLatent(data, latentModel, discreteLatentVariables);
    }

    public static DataOnMemory<DataInstance> completeDiscreteLatent(DataOnMemory<DataInstance> data, BayesianNetwork latentModel, Variable discreteLatentVariable) {
        List<Variable> discreteLatentVariables = new ArrayList<>(1);
        discreteLatentVariables.add(discreteLatentVariable);

        return completeDiscreteLatent(data, latentModel, discreteLatentVariables);
    }

    /**
     * Establishses the max and min values of the attributes using the data (it used in GenieWriter).
     */
    public static void defineAttributesMaxMinValues(DataOnMemory<DataInstance> data) {
        for(Attribute attribute: data.getAttributes()){
            if(attribute.isContinuous()) {
                /* Project data */
                List<Attribute> attributes = new ArrayList<>();
                attributes.add(attribute);
                DataOnMemory<DataInstance> projectedData = project(data, attributes);
                double[] projectedDataArray = new double[projectedData.getNumberOfDataInstances()];
                for(int i = 0; i < projectedData.getNumberOfDataInstances(); i++)
                    projectedDataArray[i] = projectedData.getDataInstance(i).toArray()[0];

                /* Estimate min and max values */
                Tuple2<Double, Double> minMaxValues = estimateMinMaxValues(projectedDataArray);
                RealStateSpace stateSpace = attribute.getStateSpaceType();
                stateSpace.setMinInterval(minMaxValues.getFirst());
                stateSpace.setMaxInterval(minMaxValues.getSecond());
            }
        }
    }

    private static DataOnMemory<DataInstance> completeDiscreteLatent(DataOnMemory<DataInstance> data, BayesianNetwork latentModel, List<Variable> discreteLatentVariables) {

        /* Seleccionamos las variables discretas latentes del modelo y creamos un atributo para cada una de ellas */
        List<Attribute> discreteLatentAttributesList = new ArrayList<>(discreteLatentVariables.size());
        int currentIndex = 0;
        for(Variable variable: discreteLatentVariables) {
            Attribute newAttribute = new Attribute(currentIndex, variable.getName(), variable.getStateSpaceType());
            discreteLatentAttributesList.add(newAttribute);
            currentIndex++;
        }

        Attributes discreteLatentAttributes = new Attributes(discreteLatentAttributesList);
        DataOnMemoryListContainer<DataInstance> completeData = new DataOnMemoryListContainer<>(discreteLatentAttributes);

        /* Iteramos por el conjunto de instancias y completamos los datos con un hard assignment de la posterior */
        for(DataInstance instance: data) {
            double[] newInstanceValues = new double[discreteLatentVariables.size()];
            for (int i = 0; i < discreteLatentVariables.size(); i++) {
                Variable latentVariable = discreteLatentVariables.get(i);
                Multinomial posterior = InferenceEngine.getPosterior(latentVariable, latentModel, instance);
                // Hard assignment (complete data with the state of maximum probability)
                int indexMaxProbability = maxProbIndex(posterior.getProbabilities());
                newInstanceValues[i] = indexMaxProbability;
            }
            completeData.add(new DataInstanceFromRawData(discreteLatentAttributes, newInstanceValues));
        }
        return completeData;
    }

    private static Tuple2<Double, Double> estimateMinMaxValues(double[] projectedDataArray) {
        double max = -Double.MAX_VALUE;
        double min = Double.MAX_VALUE;

        for(int i = 0; i < projectedDataArray.length; i++){
            if(projectedDataArray[i] > max)
                max = projectedDataArray[i];
            else if(projectedDataArray[i] < min)
                min = projectedDataArray[i];
        }

        return new Tuple2<>(min, max);
    }

    private static int maxProbIndex(double[] probs) {
        int index = 0;
        for(int i = 0; i < probs.length; i++)
            if(probs[index] < probs[i])
                index = i;

        return index;
    }
}
