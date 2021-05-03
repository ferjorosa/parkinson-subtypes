package eu.amidst.extension.util;

import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;

import java.util.*;
import java.util.stream.Collectors;

/** Samples new DAG structures using adjacency matrices */
public class DagSampler {

    private long seed;

    public DagSampler(long seed) {
        this.seed = seed;
    }

    /**
     *
     * @param baseDag Base DAG
     * @param nSamples number of new samples
     * @param maxParents max number of parents per variable
     * @param sparsityCoefficient Probability of adding a new arc [0, 1]
     * @param excludedArcs arcs that cannot be created, in addition to the user argument, it excludes by default
     *                     all arcs from continuous variables to discrete variables
     * @return the list of sampled DAGs
     */
    public List<DAG> sample(DAG baseDag,
                            int nSamples,
                            int maxParents,
                            double sparsityCoefficient,
                            Map<Integer, Set<Integer>> excludedArcs) {

        if(sparsityCoefficient > 1.0 || sparsityCoefficient < 0.0)
            throw new IllegalArgumentException("The sparsity coefficient must be between 0 and 1.0");

        Random random = new Random(seed);
        List<Variable> baseVariables = baseDag.getVariables().getListOfVariables();

        /* Generate the base adjacency matrix and the base parentCountVector */
        int[][] baseAdjacencyMatrix = generateAdjacencyMatrix(baseDag);
        int[] baseParentsCountVector = generateParentCountVector(baseAdjacencyMatrix);

        /* In addition to the user's blacklist of arcs, exclude all arcs from continuous to discrete variables. */
        List<Integer> continuousVarsIndexes = baseVariables.stream()
                .filter(Variable::isContinuous)
                .map(baseVariables::indexOf)
                .collect(Collectors.toList());
        List<Integer> discreteVarsIndexes = baseVariables.stream()
                .filter(Variable::isDiscrete)
                .map(baseVariables::indexOf)
                .collect(Collectors.toList());

        for(int contVarIndex: continuousVarsIndexes) {
            if(excludedArcs.containsKey(contVarIndex)) {
                Set<Integer> currentExcludedArcs = excludedArcs.get(contVarIndex);
                currentExcludedArcs.addAll(discreteVarsIndexes);
            }
            else
                excludedArcs.put(contVarIndex, new HashSet<>(discreteVarsIndexes));
        }

        /*
            Generate two list copies that will be shuffled for each dag sample to avoid the concentration of parents
            on the same variables (given there is a max number of parents, this would imply that certain variables would
            almost "always" have the same set of parents, especially when the number of variables is high)

            We generate two lists, one for the parent variables (i), and another for children variables (j), that can
            be represented as a directed arc [i -> j]
        */
        List<Integer> parentsList = new ArrayList<>(baseVariables.size());
        List<Integer> childrenList = new ArrayList<>(baseVariables.size());
        for(Variable var: baseDag.getVariables()) {
            parentsList.add(baseVariables.indexOf(var));
            childrenList.add(baseVariables.indexOf(var));
        }

        /* Generate DAG samples */
        List<DAG> dagSamples = new ArrayList<>(nSamples);
        for(int sample = 0; sample < nSamples; sample++) {

            boolean newDagWithCycles = true;
            DAG newDag;

            /* Repeat the process until the new dag is absent of cycles */
            do {

                /* Shuffle the parents and children (to avoid concentration of parents on the same variables) */
                Collections.shuffle(parentsList, random);
                Collections.shuffle(childrenList, random);

                /* Copy the adjacency matrix and the count vector */
                int[][] adjacencyMatrix = new int[baseAdjacencyMatrix.length][baseAdjacencyMatrix[0].length];
                for (int i = 0; i < adjacencyMatrix.length; i++)
                    adjacencyMatrix[i] = Arrays.copyOf(adjacencyMatrix[i], adjacencyMatrix[i].length);
                int[] parentsCountVector = Arrays.copyOf(baseParentsCountVector, baseParentsCountVector.length);

                /* Randomly fill the adjacency matrix */
                for (int i : parentsList)
                    for (int j : childrenList) {
                        if (i != j &&
                                (!excludedArcs.containsKey(i) || !excludedArcs.get(i).contains(j)) &&
                                parentsCountVector[j] < maxParents) {
                            if (random.nextDouble() <= sparsityCoefficient) {
                                adjacencyMatrix[i][j] = 1;
                                parentsCountVector[j] += 1;
                            }
                        }
                    }

                /* Transform the adjacency matrix into a DAG and add it to the list */
                newDag = generateDag(adjacencyMatrix, baseDag.getVariables());
                newDagWithCycles = newDag.containCycles();

            } while(newDagWithCycles);

            dagSamples.add(newDag);
        }

        return dagSamples;
    }

    private int[][] generateAdjacencyMatrix(DAG dag) {
        int[][] adjacencyMatrix = new int[dag.getVariables().getNumberOfVars()][dag.getVariables().getNumberOfVars()];
        Variable var;
        for(int varIndex = 0; varIndex < adjacencyMatrix.length; varIndex++) {
            var = dag.getVariables().getListOfVariables().get(varIndex);
            for (Variable parent : dag.getParentSet(var)) {
                int parentIndex = dag.getVariables().getListOfVariables().indexOf(parent);
                adjacencyMatrix[parentIndex][varIndex] = 1;
            }
        }
        return adjacencyMatrix;
    }

    private int[] generateParentCountVector(int[][] adjacencyMatrix) {
        int[] countVector = new int[adjacencyMatrix.length];
        for(int i = 0; i < adjacencyMatrix.length; i++)
            for(int j = 0; j < adjacencyMatrix.length; j++) {
                if(adjacencyMatrix[i][j] == 1)
                    countVector[j] += 1;
            }
        return countVector;
    }

    private DAG generateDag(int[][] adjacencyMatrix, Variables variables) {

        DAG dag = new DAG(variables);

        for(int i = 0; i < adjacencyMatrix.length; i++)
            for(int j = 0; j < adjacencyMatrix.length; j++)
                if(adjacencyMatrix [i][j] == 1)
                    dag.getParentSet(variables.getListOfVariables().get(j))
                            .addParent(variables.getListOfVariables().get(i));

        return dag;
    }
}
