package eu.amidst.extension.util;

public class LogUtils {

    public enum LogLevel {
        NONE(0),
        INFO(1),
        DEBUG(2);
        private final int value;
        LogLevel(int value) {
            this.value = value;
        }
    }

    public static void printf(String string, boolean condition) {
        if(condition)
            System.out.println(string);
    }

    public static void info(String string, LogLevel logLevel) {
        if(logLevel.value >= LogLevel.INFO.value)
            System.out.println(string);
    }

    public static void debug(String string, LogLevel logLevel) {
        if(logLevel.value >= LogLevel.DEBUG.value)
            System.out.println(string);
    }
}
