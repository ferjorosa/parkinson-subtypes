package eu.amidst.extension.util.rfmetric;

import eu.amidst.core.models.DAG;
import eu.amidst.core.models.ParentSet;
import eu.amidst.core.variables.StateSpaceTypeEnum;
import eu.amidst.core.variables.Variable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/* Un RFForest te permite calcular que particiones contiene, asi como almacenar los RFNodes*/
public class RFForest {

    List<RFNode> nodes;

    List<RFNode> roots;

    RFForest(List<RFNode> nodes, List<RFNode> roots) {
        this.nodes = nodes;
        this.roots = roots;
    }

    /** Returns all the partitions currently in the forest */
    List<RFPartition> partitions() {

        /* Recursively iterate through all the trees to fill the partitions list */
        List<RFPartition> partitions = new ArrayList<>();
        for(RFNode root: roots)
            treePartitions(root, partitions);

        return partitions;
    }

    /** Does a width first search to find all the partitions in a tree */
    private void treePartitions(RFNode node, List<RFPartition> partitions) {
        if(!node.children.isEmpty()) {
            RFPartition rfPartition = new RFPartition(node, node.children);
            partitions.add(rfPartition);

            for (RFNode child : node.children) {
                treePartitions(child, partitions);
            }
        }
    }

    public static RFForest create(DAG dag) {

        /* Check if there is a variable with more than one parent (it wouldn't be a tree) */
        for(Variable var: dag.getVariables())
            if(dag.getParentSet(var).getNumberOfParents() > 1)
                throw new IllegalArgumentException(var.getName() + " has more than one parent (it cannot be a tree)");

        /* Iterate through all the variables and create the RFNodes */
        Map<Variable, RFNode> rfNodesMap = new HashMap<>();
        List<RFNode> rfRoots = new ArrayList<>();
        for(Variable var: dag.getVariables()) {
            RFNode.VarSpace varSpace = RFNode.VarSpace.DISCRETE;
            if(var.getStateSpaceTypeEnum() == StateSpaceTypeEnum.REAL)
                varSpace = RFNode.VarSpace.CONTINUOUS;

            RFNode rfNode;
            if(var.isObservable())
                rfNode = new RFNode(var.getName(), RFNode.VarType.MANIFEST, varSpace, new ArrayList<>(), var.getNumberOfStates());
            else
                rfNode = new RFNode("LV", RFNode.VarType.LATENT, varSpace, new ArrayList<>(), var.getNumberOfStates());

            rfNodesMap.put(var, rfNode);
        }

        /* Iterate through all the parent sets to set the nodes' children */
        for(ParentSet parentSet: dag.getParentSets()) {
            if(parentSet.getNumberOfParents() > 0) {
                RFNode parentNode = rfNodesMap.get(parentSet.getParents().get(0));
                RFNode childNode = rfNodesMap.get(parentSet.getMainVar());
                parentNode.children.add(childNode);
            }
        }

        /* Iterate through all the variables to define the roots in the RFForest (those vars with children but no parent) */
        for(Variable var: dag.getVariables()){
            RFNode rfNode = rfNodesMap.get(var);
            if(dag.getParentSet(var).getNumberOfParents() == 0 && !rfNode.children.isEmpty())
                rfRoots.add(rfNode);
        }

        List<RFNode> rfNodeList = new ArrayList<>(rfNodesMap.values());

        /* Special case: create the "Empty" node, which will be the parent of all the independent nodes */
        RFNode emptyRfNode = new RFNode("RF_EMPTY", RFNode.VarType.NONE, RFNode.VarSpace.NONE, new ArrayList<>(), -1);
        for(Variable var: dag.getVariables()){
            RFNode rfNode = rfNodesMap.get(var);
            if(dag.getParentSet(var).getNumberOfParents() == 0 && rfNode.children.isEmpty()) // Node is independent
                emptyRfNode.children.add(rfNode);
        }
        rfNodeList.add(emptyRfNode);
        rfRoots.add(emptyRfNode);

        return new RFForest(rfNodeList, rfRoots);
    }
}
