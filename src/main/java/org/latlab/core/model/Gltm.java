package org.latlab.core.model;

import org.latlab.core.util.Variable;

import java.util.Collection;

/**
 * Represents a generalized latent tree model.
 * 
 * @author leonard
 * 
 */
public class Gltm extends TreeModel {

    /**
     * Constructs a local independence model with the given observed {@code
     * variables} as leaf nodes.
     * 
     * @param variables
     *            observed variables
     * @return a local independence model
     */
    public static Gltm constructLocalIndependenceModel(
        Collection<? extends Variable> variables) {
        return Builder.buildLocalIndependenceModel(
            new Gltm(), variables);
    }

    public Gltm(String name) {
        super(name);
    }

    protected Gltm(Gltm other) {
        super(other);
    }

    public Gltm() {}

    public Gltm clone() {
        return new Gltm(this);
    }
}
