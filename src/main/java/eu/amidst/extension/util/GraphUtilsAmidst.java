package eu.amidst.extension.util;

import eu.amidst.core.models.DAG;
import eu.amidst.core.models.ParentSet;
import eu.amidst.core.variables.Variable;

import java.util.*;

/**
 * - Depth First Search (DFS)
 * - Exists Path between two nodes in the graph
 * - Topological Sort
 * - Children of variable
 *
 * These methods are very inefficient because the AMIDST Dag class is not designed for them. It is better to use the Voltric
 * Dag class. For example, AMIDST Dag does not consider child nodes so we have to estimate them using their parent sets.
 */
public class GraphUtilsAmidst {

    public static List<Variable> getChildren(Variable variable, DAG dag) {

        if(!dag.getVariables().getListOfVariables().contains(variable))
            throw new IllegalArgumentException("Variable doesn't belong to the graph");

        List<Variable> children = new ArrayList<>();

        /* Iteramos por el conjunto de los ParentSets, si contiene a la variable objetivo, es que es su hijo */
        for(ParentSet parentSet: dag.getParentSets()){
            if(parentSet.contains(variable))
                children.add(parentSet.getMainVar());
        }

        return children;
    }

    public static List<Variable> getObservedChildren(Variable variable, DAG dag) {
        if(!dag.getVariables().getListOfVariables().contains(variable))
            throw new IllegalArgumentException("Variable doesn't belong to the graph");

        List<Variable> children = new ArrayList<>();

        /* Iteramos por el conjunto de los ParentSets, si contiene a la variable objetivo, es que es su hijo */
        for(ParentSet parentSet: dag.getParentSets()){
            if(parentSet.contains(variable) && parentSet.getMainVar().getAttribute() != null)
                children.add(parentSet.getMainVar());
        }

        return children;
    }

    /**
     * AMIDST no tiene programado el concepto de nodos hijo, por lo que es necesario generar un Map con los hijos de cada
     * nodo. Dicho Map sera necesario para hacer DFS.
     */
    public static Map<Variable, List<Variable>> computeChildrenMap(DAG dag) {
        Map<Variable, List<Variable>> childrenMap = new HashMap<>();

        /* Inicializamos el conjunto de hijos de cada nodo con una lista vacia */
        for(Variable node: dag.getVariables())
            childrenMap.put(node, new ArrayList<>());

        for(Variable node: dag.getVariables()) {
            ParentSet parents = dag.getParentSet(node);

            for(Variable parent: parents.getParents()) {
                List<Variable> children = childrenMap.get(parent);
                children.add(node);
                childrenMap.put(parent, children);
            }
        }

        return childrenMap;
    }

    public static void dfs(Variable node, List<Variable> visitedNodes, Map<Variable, List<Variable>> childrenMap) {

        if(!childrenMap.containsKey(node))
            throw new IllegalArgumentException("The graph must contain the argument node");

        visitedNodes.add(node);

        // explores unvisited children
        for (Variable child :  childrenMap.get(node)) {
            if (!visitedNodes.contains(child)) {
                dfs(child, visitedNodes,childrenMap);
            }
        }
    }

    /**
     * Note: No es necesario ponerle una excepcion en caso de que no pertenezcan ya que no encontraria camino o saltaria
     * excepcion en la DFS
     */
    public static boolean containsPath(Variable start, Variable end, DAG dag) {

        List<Variable> visitedNodes = new ArrayList<>();
        Map<Variable, List<Variable>> childrenMap = GraphUtilsAmidst.computeChildrenMap(dag);

        // DFS
        dfs(start, visitedNodes, childrenMap);

        // returns true if the end has been discovered
        return visitedNodes.contains(end);
    }

    public static List<Variable> topologicalSort(DAG dag) {

        Deque<Variable> sortedNodes = new LinkedList<>(); // JDK Queue
        List<Variable> visitedNodes = new ArrayList<>();
        Map<Variable, List<Variable>> childrenMap = GraphUtilsAmidst.computeChildrenMap(dag);

        /* Iteramos por el conjunto de nodos del grafo, el punto de inicio es indiferente para el algoritmo */
        for(Variable node: dag.getVariables()){
            if(!visitedNodes.contains(node))
                dfsForTopologicalSort(node, visitedNodes, sortedNodes, childrenMap);
        }

        return (List<Variable>) sortedNodes; // Given its underlying is a LinkedList, we simply cast it to List
    }

    private static void dfsForTopologicalSort(Variable node, List<Variable> visitedNodes, Deque<Variable> sortedNodes, Map<Variable, List<Variable>> childrenMap) {

        visitedNodes.add(node);

        // explores unvisited children
        for (Variable child :  childrenMap.get(node)) {
            if (!visitedNodes.contains(child)) {
                dfsForTopologicalSort(child, visitedNodes, sortedNodes, childrenMap);
            }
        }

        // Lo añadimos al final del metodo para que este la lista correctamente ordenada
        sortedNodes.addFirst(node);
    }
}
