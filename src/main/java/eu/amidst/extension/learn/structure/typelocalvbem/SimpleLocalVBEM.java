package eu.amidst.extension.learn.structure.typelocalvbem;

import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.extension.util.GraphUtilsAmidst;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Before applying local EM we have to select which nodes are going to be updated. In this case we will update the
 * argument variables and their respective children.
 * */
public class SimpleLocalVBEM implements TypeLocalVBEM {

    @Override
    public Set<Variable> variablesToUpdate(List<Variable> variables, DAG dag) {

        Set<Variable> variablesToUpdate = new LinkedHashSet<>();

        for(Variable variable: variables) {
            variablesToUpdate.add(variable);
            variablesToUpdate.addAll(GraphUtilsAmidst.getChildren(variable, dag));
        }

        return variablesToUpdate;
    }
}
