package org.latlab.core.learner.geast;

import org.latlab.core.data.MixedDataSet;
import org.latlab.core.io.bif.BifWriter;
import org.latlab.core.learner.geast.operators.SearchCandidate;

import java.io.*;
import java.nio.channels.FileLock;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Used to log the progress of GEAST. It uses lazy creation, so that the files
 * and directory are not created until they are being written to.
 * 
 * @author leonard
 * 
 */
public class Log {

	private final String baseDirectory;
	private final String directorySuffix;

	private String directory;
	private PrintWriter writer;
	private final static DateFormat FILE_DATE_FORMAT =
			new SimpleDateFormat("yyyyMMdd-HHmmss");
	private boolean directoryCreated = false;

	public final static DateFormat LOG_DATE_FORMAT =
			new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

	private Date startTime = null;
	private int majorNumber = 0;
	private int minorNumber = -1; // set to -1 so that it will become zero even
	// without increment

	private static PrintWriter errorWriter = new PrintWriter(System.out, true);

	private static File tmpDir = null;

	public static void setErrorWriter(OutputStream stream) {
		errorWriter = new PrintWriter(stream, true);
	}

	public static PrintWriter errorWriter() {
		return errorWriter;
	}

	public Log(String directory, String suffix) {
		suffix = suffix == null ? "" : "-" + suffix;
		this.baseDirectory = directory;
		this.directorySuffix = suffix;
	}

	public static void setTmpDir(String path) throws IOException {
		tmpDir = new File(path);
		if (!tmpDir.exists() && !tmpDir.mkdirs()) {
			throw new IOException("Cannot create tmp directory: " + path);
		}
	}

	public void close() {
		if (writer != null)
			writer.close();
	}

	private String write(String name, IModelWithScore estimation) {
		createDirectory();

		String filename = directory + File.separator + name;
		try {
			BifWriter writer = new BifWriter(new FileOutputStream(filename));
			writer.write(estimation);
		} catch (Exception e) {
			errorWriter().println("Error in writing file: " + filename);
			write(e);

			try {
				// try to write the string representation if writing a BIF
				// file has failed
				PrintWriter writer = new PrintWriter(filename);
				writer.write(estimation.model().toString());
				writer.close();
			} catch (Exception e1) {
				// ignore
			}
		}

		return name;
	}

	public PrintWriter writer() {
		if (writer == null) {
			String filename = null;
			try {
				createDirectory();
				filename = directory + File.separator + "log.xml";
				OutputStreamWriter osw =
						new OutputStreamWriter(new FileOutputStream(filename),
								"UTF-8");
				writer = new PrintWriter(osw, true);
			} catch (Exception e) {
				errorWriter().println("Error creating log file: " + filename);
				write(e);
				writer = errorWriter();
			}
		}

		return writer;
	}

	private String generateDirectoryName() {
		return baseDirectory + File.separator
				+ FILE_DATE_FORMAT.format(new Date()) + directorySuffix;
	}

	private void createDirectory() {
		if (directoryCreated)
			return;

		File base = new File(baseDirectory);
		if (!base.exists() && !base.mkdirs()) {
			writer = errorWriter();
			write(new Exception("Cannot create base directory: "
					+ baseDirectory));
		}

		try {
			FileOutputStream lockFile =
					new FileOutputStream(baseDirectory + File.separator
							+ "lock");
			FileLock lock = lockFile.getChannel().lock();

			// try at most five times
			for (int i = 0; i < 5 && !directoryCreated; i++) {
				directory = generateDirectoryName();
				File dir = new File(directory);

				if (dir.exists()) {
					Thread.sleep(1000);
					continue;
				}
				else if (!dir.mkdirs()) {
					throw new Exception("Unknown reason");
				}

				directoryCreated = true;
			}

			lock.release();
		} catch (Exception e) {
			writer = errorWriter();
			write(new Exception("Cannot create directory: " + directory, e));
		}
	}

	public void start() {
		if (startTime == null)
			startTime = new Date();
	}

	public Date startTime() {
		return startTime;
	}

	public long getElaspedSeconds() {
		return (System.currentTimeMillis() - startTime.getTime()) / 1000;
	}

	public void incrementMajorNumber() {
		majorNumber++;
		// set to -1 so that the next call to generateNumbers will make it
		// zero
		minorNumber = -1;
	}

	private void generateNumbers(boolean increment) {
		if (increment) {
			majorNumber++;
			minorNumber = 0;
		} else {
			minorNumber++;
		}
	}

	/**
	 * It appends a time attribute to the start element tag.
	 * 
	 * @param element
	 * @param attributes
	 */
	public void writeStartElement(String element, String attributes) {
		if (attributes == null)
			attributes = "";

		writer().format("<%s name= '%s' %s time='%d'>", element, attributes,
				getElaspedSeconds());
		writer().println();
	}

	/**
	 * Logs an element with additional attributes derived from the {@code
	 * candidate}.
	 * 
	 * @param element
	 *            name of XML element to log
	 * @param candidate
	 *            from which additional attribute values are derived
	 * @param useOrigin
	 *            whether to use the origin name instead of the model name of
	 *            the estimation
	 */
	public void writeElement(String element, SearchCandidate candidate,
			boolean useOrigin) {

		IModelWithScore estimation = candidate.estimation();
		String modelName =
				useOrigin ? estimation.origin().getName()
						: estimation.model().getName();

		writer().format(
				"<%s model='%s' score='%f' op='%s' %s bic='%f'"
						+ " loglikelihood='%f' time='%d'/>", element,
				modelName, candidate.score(), candidate.operatorName(),
				candidate.attributes(), estimation.BicScore(),
				estimation.loglikelihood(), getElaspedSeconds());
		writer().println();
	}

	/**
	 * Logs an element with attribute values derived from the {@code estimation}
	 * , and the {@code estimation} is written to a file.
	 * 
	 * @param element
	 *            name of XML element to log
	 * @param estimation
	 *            estimation to log
	 * @param info
	 *            info string used in the file name
	 * @param increment
	 *            whether to increment the major number
	 */
	public void writeElementWithEstimationToFile(String element,
			IModelWithScore estimation, String info, boolean increment) {
		generateNumbers(increment);

		String filename = write(estimation, info);

		writer().format(
				"<%s name= '%s' origin='%s' file='%s' bic='%.4f' "
						+ "loglikelihood='%.4f' time='%d'/>", element,
				estimation.model().getName(), estimation.origin().getName(),
				filename, estimation.BicScore(), estimation.loglikelihood(),
				getElaspedSeconds());
		writer().println();
	}

	/**
	 * Logs an element and writes the corresponding candidate to file.
	 * 
	 * @param element
	 *            name of XML element to log
	 * @param candidate
	 *            corresponding candidate
	 * @param increment
	 *            whether to increment the major number
	 */
	public void writeElementWithCandidateToFile(String element,
			SearchCandidate candidate, boolean increment) {
		writeElementWithEstimationToFile(element, candidate.estimation(),
				candidate.name(), increment);
	}

	/**
	 * Write the {@code estimation} to a file.
	 * 
	 * <p>
	 * Suppose the major number is 42 and minor number is 3. If info is {@code
	 * test}, then the name of the file is {@code model.0042-03.test.bif}. If
	 * the info is {@code null} or has zero length, the name of file is {@code
	 * model.0042-03.bif}.
	 * 
	 * @param estimation
	 *            estimation to write
	 * @param info
	 *            an information string used in the file name
	 * @return name of the file written to
	 */
	private String write(IModelWithScore estimation, String info) {

		String name;
		if (info != null && info.length() > 0) {
			name =
					String.format("model.%04d-%02d.%s.bif", majorNumber,
							minorNumber, info);
		} else {
			name =
					String.format("model.%04d-%02d.bif", majorNumber,
							minorNumber, info);
		}

		write(name, estimation);

		return name;
	}

	public void write(Exception exception) {
		writer().format("<exception name='%s' message='%s'>",
				exception.getClass().getSimpleName(), exception.getMessage());
		writer().println();
		writer().println("<![CDATA[");
		exception.printStackTrace(writer());
		writer().println("]]>");
		writer().println("</exception>");
	}

	public void write(Exception exception, IModelWithScore current,
			String attributes) {
		if (current == null) {
			write(exception);
			return;
		}

		if (attributes == null) {
			attributes = "";
		}

		generateNumbers(false);
		String filename = write(current, "exception");

		writer().format(
				"<exception name='%s' message='%s' model='%s' file='%s' %s>",
				exception.getClass().getSimpleName(), exception.getMessage(),
				current.model().getName(), filename, attributes);
		writer().println();
		writer().println("<![CDATA[");
		exception.printStackTrace(writer());
		writer().println("]]>");
		writer().println("</exception>");
	}

	public void write(Exception exception, SearchCandidate candidate) {
		if (candidate == null) {
			write(exception);
			return;
		}

		IModelWithScore estimation = candidate.estimation();
		if (estimation == null)
			estimation = new ModelWithoutScore(candidate.model());

		write(exception, estimation, candidate.attributes());
	}

	public void writeStartElementWithTime(String element,
			String otherAttributesString) {
		if (otherAttributesString == null) {
			writer().format("<%s time='%d'>", element, getElaspedSeconds());
		} else {
			writer().format("<%s time='%d' %s>", element, getElaspedSeconds(),
					otherAttributesString);

		}

		writer().println();
	}

	public void writeEndElement(String element) {
		writer().format("</%s>", element);
		writer().println();
	}

	public static String writeTemporaryFile(String prefix, Estimation estimation) {
		OutputStream stream;
		String path = "";

		try {
			File file = File.createTempFile(prefix, ".bif", tmpDir);
			stream = new FileOutputStream(file);
			path = file.getCanonicalPath();
		} catch (IOException e) {
			e.printStackTrace();
			stream = System.out;
			path = "System.out";
		}

		try {
			BifWriter writer = new BifWriter(stream);
			writer.write(estimation);
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}

		return path;
	}

	public void writeDataElement(MixedDataSet data) {
		String classVariableName =
				data.getClassVariable() == null ? "none"
						: data.getClassVariable().getName();
		writer.format("<data name='%s' filename='%s' class='%s'/>",
				data.name(), data.filename(), classVariableName);
		writer.println();
	}

	public void writeCommandLine(String commandLine) {
		if (commandLine == null) {
			commandLine = "[Not given]";
		}

		writer.format("<command line='%s' />", commandLine);
		writer.println();
	}
}
