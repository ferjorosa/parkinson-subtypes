package org.latlab.core.data.io;

import java.io.*;

public abstract class BaseWriter {
    protected static String defaultEncoding = "UTF-8";

    protected static OutputStreamWriter createWriter(String name)
        throws UnsupportedEncodingException, FileNotFoundException {
        return new OutputStreamWriter(
            new FileOutputStream(name), defaultEncoding);
    }

    protected static OutputStreamWriter createWriter(OutputStream output)
        throws UnsupportedEncodingException, FileNotFoundException {
        return new OutputStreamWriter(output, defaultEncoding);
    }
}
