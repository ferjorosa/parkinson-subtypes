package org.latlab.core.util;

/**
 * Combines two converters into a single converter.
 * 
 * @author leonard
 * 
 * @param <Original>
 *            original type
 * @param <Intermediate>
 *            intermediate type
 * @param <Target>
 *            target type
 */
public class ConverterComposite<Original, Intermediate, Target>
    implements Converter<Original, Target> {

    private final Converter<Original, Intermediate> first;
    private final Converter<Intermediate, Target> second;

    public ConverterComposite(
        Converter<Original, Intermediate> first,
        Converter<Intermediate, Target> second) {
        this.first = first;
        this.second = second;
    }

    public Target convert(Original o) {
        return second.convert(first.convert(o));
    }
}
