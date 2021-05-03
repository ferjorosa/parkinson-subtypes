package eu.amidst.extension.util.mi;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.variables.Variable;
import eu.amidst.extension.util.distance.DistanceFunction;

import java.util.List;
import java.util.Random;

/**
 * Clase general cuyo metodo redirige a la implementacion adecuada segun el tipo de los atributos:
 * - Continuous & Continuous -> ApproximateMI.cc()
 * - Continuous & Discrete   -> AproximateMI.cd()
 * - Discrete & Discrete     -> DiscreteMI.dd()
 *
 * When both attributes/variables are the same, its MI is assigned as 0.
 *
 * Normalization is done with the minimum of their entropies, which is the closes lower bound of the MI.
 */
public class MutualInformation {

    public static double[][] estimate(DataOnMemory<DataInstance> data, int n_neighbors, DistanceFunction distanceFunction, boolean gaussianNoise, long gaussianNoiseSeed, boolean normalization) {

        List<Attribute> attributes = data.getAttributes().getFullListOfAttributes();
        double[][] mis = new double[attributes.size()][attributes.size()];
        for(int x=0; x < attributes.size(); x++) {
            for(int y = x + 1; y < attributes.size(); y++) {

                Attribute attributeX = attributes.get(x);
                Attribute attributeY = attributes.get(y);

                //System.out.println("Attributes: (" + attributeX.getName() + "," + attributeY.getName() + ")");

                /* Projection of the dataSet in raw form */
                double[][] rawProjectedData = new double[data.getNumberOfDataInstances()][2];
                for(int i = 0; i < rawProjectedData.length; i++){
                    rawProjectedData[i][0] = data.getDataInstance(i).getValue(attributeX);
                    rawProjectedData[i][1] = data.getDataInstance(i).getValue(attributeY);
                }
                /* Estimate the MI */
                double mi = MutualInformation.estimate(rawProjectedData, attributeX, attributeY, n_neighbors, distanceFunction, gaussianNoise, gaussianNoiseSeed, normalization);
                mis[x][y] = mi;
                mis[y][x] = mi;
            }
        }

        return mis;
    }

    public static double estimate(double[][] data, Attribute x, Attribute y, int n_neighbors, DistanceFunction distanceFunction, boolean gaussianNoise, long gaussianNoiseSeed, boolean normalization) {

        if(x.equals(y)) // When both attributes/variables are the same, its MI is assigned as 0.
            return 0;

        else if(x.isDiscrete() && y.isDiscrete())
            return DiscreteMI.dd(data, x, y, normalization);

        else if(x.isDiscrete() && y.isContinuous()) {

            if(gaussianNoise)
                addGaussianNoise(data, new int[]{1}, new Random(System.nanoTime()));

            return ApproximateMI.cd(data, n_neighbors, x, y, distanceFunction);
        }

        else if(x.isContinuous() && y.isDiscrete()) {

            if(gaussianNoise)
                addGaussianNoise(data, new int[]{0}, new Random(System.nanoTime()));

            /* Swap X & Y columns in data to match the method expected order */
            double[][] swappedData = new double[data.length][2];
            for(int i = 0; i < swappedData.length; i++){
                swappedData[i][0] = data[i][1];
                swappedData[i][1] = data[i][0];
            }
            return ApproximateMI.cd(swappedData, n_neighbors, y, x, distanceFunction);
        }

        else {

            if(gaussianNoise)
                addGaussianNoise(data, new int[]{0, 1}, new Random(gaussianNoiseSeed));

            return ApproximateMI.cc(data, n_neighbors, x, y, distanceFunction);
        }
    }

    public static double estimate(double[][] data, Variable x, Variable y, int n_neighbors, DistanceFunction distanceFunction, boolean gaussianNoise, long gaussianNoiseSeed, boolean normalization) {

        if(x.equals(y)) // When both attributes/variables are the same, its MI is assigned as 0.
            return 0;

        else if(x.isDiscrete() && y.isDiscrete())
            return DiscreteMI.dd(data, x, y, normalization);

        else if(x.isDiscrete() && y.isContinuous()) {

            if(gaussianNoise)
                addGaussianNoise(data, new int[]{1}, new Random(System.nanoTime()));

            return ApproximateMI.cd(data, n_neighbors, x, y, distanceFunction);
        }

        else if(x.isContinuous() && y.isDiscrete()) {

            if(gaussianNoise)
                addGaussianNoise(data, new int[]{0}, new Random(System.nanoTime()));

            /* Swap X & Y columns in data to match the method expected order */
            double[][] swappedData = new double[data.length][2];
            for(int i = 0; i < swappedData.length; i++){
                swappedData[i][0] = data[i][1];
                swappedData[i][1] = data[i][0];
            }
            return ApproximateMI.cd(swappedData, n_neighbors, y, x, distanceFunction);
        }

        else {

            if(gaussianNoise)
                addGaussianNoise(data, new int[]{0, 1}, new Random(gaussianNoiseSeed));

            return ApproximateMI.cc(data, n_neighbors, x, y, distanceFunction);
        }
    }

    private static void addGaussianNoise(double[][] data, int[] indexes, Random random) {

        for(int index: indexes)
            for(int i = 0; i < data.length; i++)
                data[i][index] = data[i][index] + 1e-10 * random.nextGaussian();

    }
}
