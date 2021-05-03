package org.latlab.core.util;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * For generating indices and default name. Each time the method {@code next} is
 * called, an index and a name are generated with the current count incremented.
 * <p>
 * The {@code encounterName} should be called when a name is encountered, so
 * that it checks to see if the given name clashes with the names that might be
 * generated.
 * 
 * @author leonard
 * 
 */
public class Counter {
    public class Instance {
        public final int index;
        public final String name;

        public Instance(int index, String name) {
            this.index = index;
            this.name = name;
        }
    }

    private int current = 0;
    private final String prefix;
    private final Pattern pattern;

    public Counter(String prefix) {
        this.prefix = prefix;
        this.pattern = Pattern.compile(prefix + "(\\d*)");
    }

    public synchronized Instance next() {
        Instance instance = new Instance(current, createName());
        current++;
        return instance;
    }

    public synchronized int nextIndex() {
        return current++;
    }

    public synchronized void encounterName(String name) {
        Matcher match = pattern.matcher(name);
        if (match.matches()) {
            int number = Integer.parseInt(match.group(1));
            if (number >= current)
                current = number + 1;
        }
    }

    private String createName() {
        return prefix + current;
    }
}
