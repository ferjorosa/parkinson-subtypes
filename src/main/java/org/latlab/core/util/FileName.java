package org.latlab.core.util;

import java.io.File;

public class FileName {
	/**
	 * Gets the the last component from the path
	 * 
	 * @param path
	 *            path
	 * @return last component
	 */
	public static String getLastComponent(String path) {
		int separatorIndex = path.lastIndexOf(File.separator);
		return path.substring(separatorIndex + 1);
	}

	/**
	 * Gets the name without extension.
	 * 
	 * @param fileName
	 *            original file name
	 * @return name without extension
	 */
	public static String getName(String fileName) {
		int index = fileName.lastIndexOf(".");
		if (index < 0)
			return fileName;

		return fileName.substring(0, index);
	}

	public static String getNameOfLastComponent(String path) {
		return getName(getLastComponent(path));
	}

	public static String getExtension(File f) {
		return getExtension(f.getName());
	}

	public static String getExtension(String name) {
		String ext = null;
		int i = name.lastIndexOf('.');

		if (i > 0 && i < name.length() - 1) {
			ext = name.substring(i + 1).toLowerCase();
		}

		return ext;
	}
}
