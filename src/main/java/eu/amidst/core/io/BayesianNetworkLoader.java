/*
 *
 *
 *    Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.
 *    See the NOTICE file distributed with this work for additional information regarding copyright ownership.
 *    The ASF licenses this file to You under the Apache License, Version 2.0 (the "License"); you may not use
 *    this file except in compliance with the License.  You may obtain a copy of the License at
 *
 *            http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software distributed under the License is
 *    distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and limitations under the License.
 *
 *
 */

package eu.amidst.core.io;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.Attributes;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.variables.Variable;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.List;
import java.util.stream.Collectors;

/**
 * This class allows to load a {@link BayesianNetwork} model from a file.
 *
 * <p> For an example of use follow this link </p>
 * <p> <a href="http://amidst.github.io/toolbox/CodeExamples.html#iobnsexample"> http://amidst.github.io/toolbox/CodeExamples.html#iobnsexample </a>  </p>
 *
 */
public final class BayesianNetworkLoader {

    /**
     * Loads a {@link BayesianNetwork} from a file.
     * @param fileName a name of the file from which the Bayesian network will be loaded.
     * @return a {@link BayesianNetwork} model.
     * @throws IOException in case of an error while reading the file.
     * @throws ClassNotFoundException in case the class is not found.
     */
    public static BayesianNetwork loadFromFile(String fileName) throws IOException, ClassNotFoundException {

        ObjectInputStream ois = new ObjectInputStream(new FileInputStream(fileName));
        Object obj = ois.readObject();
        ois.close();
        return (BayesianNetwork)obj;
    }

    /** Same function but we pass current Attributes to know which variables in the model are latent */
    public static BayesianNetwork loadFromFile(String fileName, Attributes attributes) throws IOException, ClassNotFoundException {
        ObjectInputStream ois = new ObjectInputStream(new FileInputStream(fileName));
        Object obj = ois.readObject();
        ois.close();
        BayesianNetwork bn = (BayesianNetwork) obj;

        List<String> attributeNames = attributes.getFullListOfAttributes().stream().map(Attribute::getName).collect(Collectors.toList());

        for(Variable var: bn.getVariables()) {
            if(!attributeNames.contains(var.getName())) {
                var.setAttribute(null);
                var.setObservable(false);
            }
        }

        return bn;
    }
}

