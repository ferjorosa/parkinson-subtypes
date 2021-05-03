package eu.amidst.extension.learn.parameter.penalizer;

import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.extension.util.MyMath;

/**
 * Este penalizdor propuesto en la seccion 10.2.4 del capitulo de Bishop penaliza el ELBO con respecto a la cardinalidad
 * de sus variables latentes discretas. Lo hemos extendido para multiples variables latentes:
 *
 * ELBO_{penalized} = ELBO - \sum_{i} log K_{i}!        -> "i" representa el iterador de las variables latentes
 */
public class BishopPenalizer implements ElboPenalizer {

    @Override
    public double penalize(double elbo, DAG dag) {

        /* Iteramos por el DAG para obtener las cardinalidades de las variables latentes discretas */
        double penalizer = 0;
        for(Variable variable: dag.getVariables()){
            if(!variable.isObservable() && variable.isDiscrete()) {
                long fact = MyMath.factorial(variable.getNumberOfStates());
                penalizer += Math.log(fact);
            }
        }

        return elbo - penalizer;
    }
}
