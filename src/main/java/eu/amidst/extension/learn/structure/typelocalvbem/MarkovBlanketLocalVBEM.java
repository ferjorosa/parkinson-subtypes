package eu.amidst.extension.learn.structure.typelocalvbem;

import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.extension.util.GraphUtilsAmidst;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Before applying local EM we have to select which nodes are going to be updated. In this case we will update the
 * argument variables and their combined (overlapping) markov blanket
 * */
public class MarkovBlanketLocalVBEM implements TypeLocalVBEM {

    @Override
    public Set<Variable> variablesToUpdate(List<Variable> variables, DAG dag) {

        Set<Variable> variablesToUpdate = new LinkedHashSet<>();

        for(Variable variable: variables) {

            variablesToUpdate.add(variable);

            for(Variable child: GraphUtilsAmidst.getChildren(variable, dag)){
                variablesToUpdate.add(child);
                variablesToUpdate.addAll(dag.getParentSet(child).getParents());
            }
        }

        return variablesToUpdate;
    }
}
