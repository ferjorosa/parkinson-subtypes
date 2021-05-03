package org.latlab.core.util;

import java.io.PrintWriter;

public class BuildConfig {
    public static final boolean PROFILE = false;

    private static PrintWriter profileWriter;

    public static void init() {
        try {
            profileWriter = PROFILE ? new PrintWriter("profile.log.txt") : null;
        } catch (Exception e) {
            e.printStackTrace();
            profileWriter = null;
        }
    }

    public static void profileLog(String message) {
        if (PROFILE)
            profileWriter.println(message);
    }

    public static void cleanup() {
        if (PROFILE) {
            profileWriter.close();
            profileWriter = null;
        }
    }
}
