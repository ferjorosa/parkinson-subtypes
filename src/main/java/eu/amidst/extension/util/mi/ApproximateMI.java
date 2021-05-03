package eu.amidst.extension.util.mi;


import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.variables.Variable;
import eu.amidst.extension.util.distance.DistanceFunction;
import eu.amidst.extension.util.neighborsearch.KDTree;
import eu.amidst.extension.util.neighborsearch.Neighbor;
import org.apache.commons.math3.special.Gamma;

import java.util.ArrayList;
import java.util.List;

/**
 * Esta clase esta pensada para el calculo de MI entre 2 variables continuas o 1 variable continua y 1 discreta.
 * Se calcula de forma aproximada mediante Nearest Neighbours.
 */
class ApproximateMI {

    public static double cc(double[][] data, int n_neighbors, Attribute contAttrX, Attribute contAttrY, DistanceFunction distanceFunction) {
        if(!contAttrX.isContinuous() || !contAttrY.isContinuous())
            throw new IllegalArgumentException("Both attributes have to be continuous");

        return cc(data, n_neighbors, distanceFunction);
    }

    public static double cc(double[][] data, int n_neighbors, Variable contVarX, Variable contVarY, DistanceFunction distanceFunction) {
        if(!contVarX.isContinuous() || !contVarY.isContinuous())
            throw new IllegalArgumentException("Both variables have to be continuous");

        return cc(data, n_neighbors, distanceFunction);
    }

    public static double cd(double[][] data, int n_neighbors, Attribute discAttrX, Attribute contAttrY, DistanceFunction distanceFunction) {
        if(!contAttrY.isContinuous() || !discAttrX.isDiscrete())
            throw new IllegalArgumentException("discAttrX has to be discrete and contAttrY has to be continuous");

        return cd(data, discAttrX.getNumberOfStates(), n_neighbors, distanceFunction);
    }

    public static double cd(double[][] data, int n_neighbors, Variable discVarX, Variable contVarY, DistanceFunction distanceFunction) {
        if(!contVarY.isContinuous() || !discVarX.isDiscrete())
            throw new IllegalArgumentException("discVarX has to be discrete and contVarY has to be continuous");

        return cd(data, discVarX.getNumberOfStates(), n_neighbors, distanceFunction);
    }

    private static double cc(double[][] data, int n_neighbors, DistanceFunction distanceFunction) {

        /** Note for developers: Utilizamos la notacion del articulo para el nombramiento de las variables */

        /* 0 - For faster computation we use 2 1D vectors for X,Y subspaces */
        int N = data.length;
        double[] dataX = new double[N];
        for(int i = 0; i < N; i++)
            dataX[i] = data[i][0];

        double[] dataY = new double[N];
        for(int i = 0; i < N; i++)
            dataY[i] = data[i][1];

        /* 1 - Genero el KDTree */
        KDTree<double[]> kdTree = new KDTree<>(data, data, distanceFunction, true);

        /* 2 - Itero por cada uno de los puntos y obtengo nx(i) & ny(i) */
        double digammaXsum = 0;
        double digammaYsum = 0;
        for(int i=0; i < N; i++) {

            Neighbor<double[], double[]>[] neighbors = kdTree.knn(data[i], n_neighbors);
            Neighbor<double[], double[]> kNeighbor = neighbors[0]; // The k-nearest-neighbor is the first in the array
            double eX = distanceFunction.distance(kNeighbor.value[0], data[i][0]); // Absolute distance in the X dimension
            double eY = distanceFunction.distance(kNeighbor.value[1], data[i][1]); // Absolute distance in the Y dimension
            double e = Math.max(eX, eY);

            int nX = countPointsInRange(dataX, data[i][0], e, distanceFunction);
            int nY = countPointsInRange(dataY, data[i][1], e, distanceFunction);

            digammaXsum += Gamma.digamma(nX + 1);
            digammaYsum += Gamma.digamma(nY + 1);
        }

        double gamma_N = Gamma.digamma(N);
        double gamma_k = Gamma.digamma(n_neighbors);
        double gamma_nx = digammaXsum / N;
        double gamma_ny = digammaYsum / N;

        return Math.max(0, gamma_N + gamma_k - gamma_nx - gamma_ny);
    }

    private static double cd(double[][] data, int nDiscreteStates, int n_neighbors, DistanceFunction distanceFunction) {

        /** Note for developers: Utilizamos la notacion del articulo para el nombramiento de las variables */

        int N = data.length;

        /* 1D vector with Y dimension data (it is required to estimate "m") */
        double[] dataY = new double[N];
        for(int i = 0; i < N; i++)
            dataY[i] = data[i][1];

        /* Generate a 1D dataSet in Y for each state of the categorical X (List form)*/
        List<List<Double>> dividedDataList = new ArrayList<>(nDiscreteStates);
        for(int state = 0; state < nDiscreteStates; state++) {
            dividedDataList.add(new ArrayList<>());
            for (int i = 0; i < data.length; i++) {
                if(data[i][0] == state)
                    dividedDataList.get(state).add(data[i][1]);
            }
        }

        /* Put the datasets in array form */
        List<double[][]> dividedData = new ArrayList<>();
        for(int state = 0; state < nDiscreteStates; state++) {
            List<Double> values = dividedDataList.get(state);
            double[][] valuesArray = new double[values.size()][];
            for (int i = 0; i < values.size(); i++){
                double[] Y1D = new double[]{values.get(i)};
                valuesArray[i] = Y1D;
            }
            dividedData.add(valuesArray);
        }

        /* Generate a 1D (Y dimension) KDTree for each division of data given by X */
        List<KDTree<double[]>> kdTrees = new ArrayList<>();
        for(int state = 0; state < nDiscreteStates; state++) {
            if(dividedData.get(state).length > 0) { // Ignore states with no associated Y values
                KDTree<double[]> kdTree = new KDTree<>(dividedData.get(state), dividedData.get(state), distanceFunction, true);
                // TODO: An alternative to study is to apply gaussian noise to those cases when numberOfNodes < n_neighbors (too many repetitions)
                if(kdTree.numberOfNodes() > n_neighbors) // Only consider KDtrees with an appropriate number of nodes
                    kdTrees.add(kdTree);
            }
        }

        /*  */
        double digammaYsum = 0;
        double digammaXsum = 0;
        for(int state = 0; state < kdTrees.size(); state++){

            KDTree<double[]> kdTree = kdTrees.get(state);
            double digammaX = Gamma.digamma(kdTree.numberOfNodes());

            for(int i=0; i < dividedData.get(state).length; i++) {

                Neighbor<double[], double[]>[] neighbors = kdTree.knn(dividedData.get(state)[i], n_neighbors);
                Neighbor<double[], double[]> kNeighbor = neighbors[0]; // The k nearest neighbor is the first in the array
                double eY = distanceFunction.distance(kNeighbor.value[0], dividedData.get(state)[i][0]); // Absolute distance in the Y dimension
                double e = eY; // In this case there is only the Y dimension

                double m = countPointsInRange(dataY, dividedData.get(state)[i][0], e, distanceFunction);

                digammaYsum += Gamma.digamma(m + 1);
                digammaXsum += digammaX;
            }
        }

        double gamma_N = Gamma.digamma(N);
        double gamma_k = Gamma.digamma(n_neighbors);
        double gamma_Nx = (digammaXsum / N);
        double gamma_m = (digammaYsum / N);

        return Math.max(0, gamma_N + gamma_k - gamma_Nx - gamma_m);
    }

    private static int countPointsInRange(double[] points, double point, double radius, DistanceFunction distanceFunction) {
        int count = 0;
        for(int i=0; i < points.length; i++) {
            if(distanceFunction.distance(points[i], point) < radius)
                count++;
        }
        if(count > 0)
            return count - 1; // This is because it's counting itself

        return count; // In this case occurs, for example, when radius is 0.0
    }
}
