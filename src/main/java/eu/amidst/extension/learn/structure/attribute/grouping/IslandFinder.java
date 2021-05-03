package eu.amidst.extension.learn.structure.attribute.grouping;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.data.DataUtils;
import eu.amidst.extension.learn.structure.BLTM_EAST;
import eu.amidst.extension.learn.structure.BLTM_HillClimbing;
import eu.amidst.extension.learn.structure.Result;
import eu.amidst.extension.learn.structure.operator.hc.tree.BltmHcAddNode;
import eu.amidst.extension.learn.structure.operator.hc.tree.BltmHcIncreaseCard;
import eu.amidst.extension.learn.structure.operator.hc.tree.BltmHcOperator;
import eu.amidst.extension.learn.structure.operator.hc.tree.BltmHcRelocateNode;
import eu.amidst.extension.learn.structure.typelocalvbem.TypeLocalVBEM;
import eu.amidst.extension.util.tuple.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Los nombres de las variables latentes no se tienen en cuenta ya que no hay por ahora un contador general que nos sirva
 * para poder asignarles un nombre unico
 */
public class IslandFinder {

    private TypeLocalVBEM typeLocalVBEM;

    private int maxCardinality;

    public IslandFinder(int maxCardinality, TypeLocalVBEM typeLocalVBEM) {
        this.maxCardinality = maxCardinality;
        this.typeLocalVBEM = typeLocalVBEM;
    }

    public DAG generate(DataOnMemory<DataInstance> data, double[][] mutualInformations, Map<String, double[]> priors) {

        Set<Integer> attributeIndexSet = new LinkedHashSet<>();
        for(int i = 0; i < data.getAttributes().getNumberOfAttributes(); i++)
            attributeIndexSet.add(i);

        List<Tuple2<List<Integer>, Integer>> islands = new ArrayList<>();

        /* Seguimos iterando mientras queden mas de 2 atributos en la lista*/
        while(attributeIndexSet.size() >= 2) {

            /* Escogemos el par de atributos con mayor MI y los eliminamos de la lista*/
            List<Integer> groupIndexes = new ArrayList<>();
            Tuple2<Integer, Integer> highestMIattributesIndexes = highestMIindexes(attributeIndexSet, mutualInformations);
            groupIndexes.add(highestMIattributesIndexes.getFirst());
            groupIndexes.add(highestMIattributesIndexes.getSecond());
            attributeIndexSet.remove(highestMIattributesIndexes.getFirst());
            attributeIndexSet.remove(highestMIattributesIndexes.getSecond());

            boolean tryToAddNodesToGroup = false;
            int islandLatentVarCardinality = -1;
            do {

                /* Obtenemos el atributo mas cercano al grupo */
                int closestAttributeToTheGroup = getClosestAttributeToTheGroup(groupIndexes, attributeIndexSet, mutualInformations);
                if(closestAttributeToTheGroup != -1) {

                    /* UD Test */
                    Tuple2<Boolean, Integer> udTest = unidimensionalityTest(groupIndexes, closestAttributeToTheGroup, data, priors);
                    if (udTest.getFirst()) {
                        groupIndexes.add(closestAttributeToTheGroup);
                        attributeIndexSet.remove(closestAttributeToTheGroup);
                        tryToAddNodesToGroup = true;
                        islandLatentVarCardinality = udTest.getSecond();
                    } else
                        tryToAddNodesToGroup = false;
                }

            } while (tryToAddNodesToGroup && attributeIndexSet.size() > 0);

            if(islandLatentVarCardinality == -1)
                islandLatentVarCardinality = learnCardinality(attributeIndexSet, data,  priors);

            islands.add(new Tuple2<>(groupIndexes, islandLatentVarCardinality));
        }

        /* Special case for when an attribute hasn't been located in any island. We move it to the closest island */
        if(attributeIndexSet.size() == 1)
            specialCase(islands, attributeIndexSet.stream().findFirst().get(), mutualInformations);

        return generateIslandsDAG(islands, data);
    }

    /**
     * Devuelve una tupla con los indices de los atributos con mayor MI
     */
    private Tuple2<Integer, Integer> highestMIindexes(Set<Integer> attributeIndexSet, double[][] mutualInformations) {
        int best_i = -1;
        int best_j = -1;
        double best_value = -1;

        /* Iteramos por la triangular superior para obtener el valor maximo */
        for(int i: attributeIndexSet)
            for(int j: attributeIndexSet){
                if(i != j)
                    if(mutualInformations[i][j] > best_value){
                        best_i = i;
                        best_j = j;
                        best_value = mutualInformations[i][j];
                    }
            }

        return new Tuple2<>(best_i, best_j);
    }

    /** Obtenemos el indice del atributo mas cercano al grupo */
    private int getClosestAttributeToTheGroup(List<Integer> groupIndexes, Set<Integer> attributeIndexSet, double[][] mutualInformations) {

        int closestAttribute = -1;
        double closestAttributeMI = -Double.MAX_VALUE;

        for(Integer i: groupIndexes) {
            for (Integer j : attributeIndexSet) {
                if (mutualInformations[i][j] > closestAttributeMI){
                    closestAttribute = j;
                    closestAttributeMI = mutualInformations[i][j];
                }
            }
        }
        return closestAttribute;
    }

    /** El Unidimensionality test, la idea es comprobar is existe un modelo bi-dimensional mejor que el modelo unidimensional */
    private Tuple2<Boolean, Integer> unidimensionalityTest(List<Integer> groupIndexes, int closestAttributeToTheGroup, DataOnMemory<DataInstance> data, Map<String, double[]> priors) {

        List<Attribute> attributes = new ArrayList<>();
        for(Integer i: groupIndexes)
            attributes.add(data.getAttributes().getFullListOfAttributes().get(i));
        attributes.add(data.getAttributes().getFullListOfAttributes().get(closestAttributeToTheGroup));

        DataOnMemory<DataInstance> projectedData = DataUtils.project(data, attributes);

        Result resultLCM = learnLCM(projectedData, priors);
        Result result2LTM = learn2LTM(projectedData, priors);

        /* Si no pasa el UD-Test, devolvemos false y cardinalidad -1 (simplemente para recordar que no tiene sentido considerarla)*/
        if(result2LTM.getElbo() > resultLCM.getElbo()
                && result2LTM.getDag().getVariables().getListOfVariables().size() == (attributes.size() + 2)){
            // This cardinality is only useful for the specific case of an island with only two attributes that has failed the UD-Test for the 3rd attribute
            return new Tuple2<>(false, resultLCM.getDag().getVariables().getVariableByName("LV").getNumberOfStates());
        }

        /* Si pasa el UD-Test devolvemos true y la cardinalidad de la variable latente del LCM */
        return new Tuple2<>(true, resultLCM.getDag().getVariables().getVariableByName("LV").getNumberOfStates());
    }

    private Result learnLCM(DataOnMemory<DataInstance> data, Map<String, double[]> priors) {

        /* Creamos un Naive Bayes con padre latente */
        Variables variables = new Variables(data.getAttributes());
        Variable latentVar = variables.newMultinomialVariable("LV", 2);

        DAG lcmDag = new DAG(variables);

        for(Variable var: variables)
            if(!var.equals(latentVar))
                lcmDag.getParentSet(var).addParent(latentVar);

        Set<BltmHcOperator> operators = new LinkedHashSet<>();
        operators.add(new BltmHcIncreaseCard(this.maxCardinality, this.typeLocalVBEM));
        BLTM_HillClimbing hillClimbing = new BLTM_HillClimbing(operators, false);

        return hillClimbing.learnModel(lcmDag, data, priors,false);
    }

    private Result learn2LTM(DataOnMemory<DataInstance> data, Map<String, double[]> priors) {

        /* Creamos un Naive Bayes con padre latente */
        Variables variables = new Variables(data.getAttributes());
        Variable latentVar = variables.newMultinomialVariable("LV", 2);

        DAG lcmDag = new DAG(variables);

        for(Variable var: variables)
            if(!var.equals(latentVar))
                lcmDag.getParentSet(var).addParent(latentVar);

        /* Aprendemos el mejor modelo con un maximo de 2 variables latentes */
        Set<BltmHcOperator> expansionOperators = new LinkedHashSet<>();
        Set<BltmHcOperator> simplificationOperators = new LinkedHashSet<>();
        Set<BltmHcOperator> adjustmentOperators = new LinkedHashSet<>();
        expansionOperators.add(new BltmHcIncreaseCard(this.maxCardinality, typeLocalVBEM));
        expansionOperators.add(new BltmHcAddNode(2, typeLocalVBEM));
        adjustmentOperators.add(new BltmHcRelocateNode(typeLocalVBEM));

        BLTM_EAST east = new BLTM_EAST(
                expansionOperators,
                simplificationOperators,
                adjustmentOperators,
                false);

        return east.learnModel(lcmDag, data, priors,false);
    }

    private int learnCardinality(Set<Integer> attributeIndexSet, DataOnMemory<DataInstance> data, Map<String, double[]> priors) {

        /* Project data */
        List<Attribute> attributes = attributeIndexSet.stream()
                .map(x->data.getAttributes().getFullListOfAttributes().get(x)).collect(Collectors.toList());

        DataOnMemory<DataInstance> projectedData = DataUtils.project(data, attributes);

        /* Learn LCM cardinality */
        Variables variables = new Variables(projectedData.getAttributes());
        Variable latentVar = variables.newMultinomialVariable("LV", 2);

        DAG dag = new DAG(variables);

        for(Variable var: variables)
            if(!var.equals(latentVar))
                dag.getParentSet(var).addParent(latentVar);

        Set<BltmHcOperator> operators = new LinkedHashSet<>();
        operators.add(new BltmHcIncreaseCard(10, typeLocalVBEM));
        BLTM_HillClimbing hillClimbing = new BLTM_HillClimbing(operators, true);
        Result result = hillClimbing.learnModel(dag, projectedData, priors,false);

        return result.getDag().getVariables().getVariableByName("LV").getNumberOfStates();
    }

    /** Special case for when an attribute hasn't been located in any island. We move it to the closest island */
    private void specialCase(List<Tuple2<List<Integer>, Integer>> islands, Integer attributeIndex, double[][] mutualInformations) {

        /* Find the closest attribute */
        int closestAttribute = -1;
        double closestAttributeMI = -Double.MAX_VALUE;

        for (int j= 1; j < mutualInformations.length; j++) {
            if (mutualInformations[attributeIndex][j] > closestAttributeMI){
                closestAttribute = j;
                closestAttributeMI = mutualInformations[attributeIndex][j];
            }
        }

        /* Find its respective island and add the attribute to it */
        for(Tuple2<List<Integer>, Integer> island: islands)
            if(island.getFirst().contains(closestAttribute))
                island.getFirst().add(attributeIndex);
    }

    private DAG generateIslandsDAG(List<Tuple2<List<Integer>, Integer>> islands, DataOnMemory<DataInstance> data) {

        /* Create the Variables object with new latent variables */
        Variables variables = new Variables();
        for(Attribute attribute: data.getAttributes())
            variables.newVariable(attribute);

        int currentLvIndex = 1;
        List<Variable> discreteLatentVariables = new ArrayList<>();
        for(Tuple2<List<Integer>, Integer> island: islands) {
            Variable lv = variables.newMultinomialVariable("LV_" + currentLvIndex, island.getSecond());
            discreteLatentVariables.add(lv);
            currentLvIndex++;
        }

        /* Create the islands DAG */
        DAG islandsDAG = new DAG(variables);

        for(int i = 0; i < islands.size(); i++) {
            Variable islandLatentVar = discreteLatentVariables.get(i);
            for (Integer attrIndex : islands.get(i).getFirst()) {
                String attributeName = data.getAttributes().getFullListOfAttributes().get(attrIndex).getName();
                islandsDAG.getParentSet(variables.getVariableByName(attributeName)).addParent(islandLatentVar);
            }
        }
        return islandsDAG;
    }
}
