package eu.amidst.extension.learn.structure.typelocalvbem;

import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/* Before applying local VBEM we have to select which nodes are going to be updated. */
public interface TypeLocalVBEM {

    Set<Variable> variablesToUpdate(List<Variable> variables, DAG dag);

    default Set<Variable> variablesToUpdate(Variable variable, DAG dag) {
        List<Variable> variables = new ArrayList<>(1);
        variables.add(variable);
        return variablesToUpdate(variables, dag);
    }

    default Set<Variable> variablesToUpdate(Variable firstVar, Variable secondVar, DAG dag) {
        List<Variable> variables = new ArrayList<>(2);
        variables.add(firstVar);
        variables.add(secondVar);
        return variablesToUpdate(variables, dag);
    }
}
