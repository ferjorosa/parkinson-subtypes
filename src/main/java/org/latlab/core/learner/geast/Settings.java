package org.latlab.core.learner.geast;

import org.latlab.core.data.MixedDataSet;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

import javax.xml.parsers.DocumentBuilderFactory;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

public class Settings {
	private final Document document;
	private final MixedDataSet data;
	private final String logDirectorySuffix;

	private final class GeastSettings {
		public int threads;
		public int screening;
		public double threshold;
	}

	/**
	 * Constructs a setting object.
	 * 
	 * @param filename
	 *            name of the setting file
	 * @param data
	 *            training data
	 * @param logDirectoryPrefix
	 *            prefix for the log directory
	 * @throws Exception
	 */
	public Settings(String filename, MixedDataSet data,
			String logDirectoryPrefix) throws Exception {
		this(filename == null ? null : new FileInputStream(filename), data,
				logDirectoryPrefix);
	}

	/**
	 * Constructs a setting object.
	 * 
	 * @param input
	 *            from which the setting is read
	 * @param data
	 *            training data
	 * @param logDirectoryPrefix
	 *            prefix for the log directory
	 * @throws Exception
	 */
	public Settings(InputStream input, MixedDataSet data,
			String logDirectoryPrefix) throws Exception {
		if (input != null) {
			document =
					DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(
							input);
		} else {
			document = null;
		}

		this.data = data;
		this.logDirectorySuffix = logDirectoryPrefix;
	}

	public EmFramework createScreenEm(CovarianceConstrainer constrainer) {
		return createEm(getEmElement("screening"), constrainer);
	}

	public EmFramework createSelectEm(CovarianceConstrainer constrainer) {
		return createEm(getEmElement("selection"), constrainer);
	}

	public EmFramework createEstimateEm(CovarianceConstrainer constrainer) {
		return createEm(getEmElement("estimation"), constrainer);
	}

	public Log createLog() throws IOException {
		Element element = getLogElement();
		Log log = new Log(element.getAttribute("path"), logDirectorySuffix);
		Log.setTmpDir(element.getAttribute("tmp"));
		return log;
	}

	/**
	 * Creates the log file in the output directory.
	 * 
	 * @return log file
	 */
	private Log createLogInOutput() {
		return new Log("output", logDirectorySuffix);
	}

	public Geast createGeast() throws IOException {
		if (document == null)
			return new Geast(1, data, createLogInOutput());

		GeastSettings settings = getGeastSetting();
		CovarianceConstrainer constrainer = createCovarianceConstrainer();
		return new Geast(settings.threads, settings.screening,
				settings.threshold, data, createLog(),
				createScreenEm(constrainer), createSelectEm(constrainer),
				createEstimateEm(constrainer));
	}

	public Geast createGeast(long seed) throws IOException {
		if (document == null)
			return new Geast(1, data, createLogInOutput());

		GeastSettings settings = getGeastSetting();
		CovarianceConstrainer constrainer = createCovarianceConstrainer();
		return new Geast(settings.threads, settings.screening,
				settings.threshold, data, createLog(),
				createScreenEm(constrainer), createSelectEm(constrainer),
				createEstimateEm(constrainer), seed);
	}

	public FmmLearner createFmmLearner(int initial, boolean increase)
			throws IOException {
		if (document == null)
			return new FmmLearner(data, createLogInOutput(), initial, increase);

		GeastSettings settings = getGeastSetting();
		CovarianceConstrainer constrainer = createCovarianceConstrainer();
		return new FmmLearner(settings.threads, settings.threshold, data,
				createLog(), createEstimateEm(constrainer), initial, increase);
	}

	public PouchMixtureModelLearner createPouchMixtureModelLearner()
			throws IOException {
		if (document == null)
			return new PouchMixtureModelLearner(data, createLogInOutput());

		GeastSettings settings = getGeastSetting();
		CovarianceConstrainer constrainer = createCovarianceConstrainer();
		return new PouchMixtureModelLearner(settings.threads,
				settings.threshold, data, createLog(),
				createEstimateEm(constrainer));
	}

	public GeastWithoutPouch createGeastWithoutPouch() throws IOException {
		if (document == null)
			return new GeastWithoutPouch(data, createLogInOutput());

		GeastSettings settings = getGeastSetting();
		CovarianceConstrainer constrainer = createCovarianceConstrainer();
		return new GeastWithoutPouch(settings.threads, settings.threshold,
				data, createLog(), createEstimateEm(constrainer));
	}

	public GeastWithoutPouch createGeastWithoutPouch(long seed) throws IOException {
		if (document == null)
			return new GeastWithoutPouch(data, createLogInOutput());

		GeastSettings settings = getGeastSetting();
		CovarianceConstrainer constrainer = createCovarianceConstrainer();
		return new GeastWithoutPouch(settings.threads, settings.threshold,
				data, createLog(), createEstimateEm(constrainer), seed);
	}

	private GeastSettings getGeastSetting() {
		if (document == null)
			return null;

		GeastSettings settings = new GeastSettings();
		Element element =
				(Element) document.getElementsByTagName("settings").item(0);

		// if the number of threads in setting file is non-positive, it sets the
		// number of threads equal to number of cores
		int threads =
				getAttributeValue(element, "threads", Geast.DEFAULT_THREADS);
		settings.threads =
				threads > 0 ? threads
						: Runtime.getRuntime().availableProcessors();
				
		settings.screening =
				getAttributeValue(element, "screening", Geast.DEFAULT_SCREENING);
		settings.threshold =
				getAttributeValue(element, "threshold",
						Geast.DEFAULT_THRESHOLD, false);

		return settings;
	}

	private double getAttributeValue(Element element, String attribute,
			double defaultValue, boolean expectedMissing) {
		try {
			if (element == null)
				throw new NullPointerException();

			String value = element.getAttribute(attribute);
			if (value.length() == 0) {
				if (expectedMissing)
					return defaultValue;
				else
					throw new IllegalArgumentException();
			}

			return Double.parseDouble(value);

		} catch (Exception e) {
			e.printStackTrace();
			System.err.println("Attribute " + attribute + " cannot be read.");
			return defaultValue;
		}
	}

	private boolean getAttributeValue(Element element, String attribute,
			boolean defaultValue, boolean expectedMissing) {
		try {
			if (element == null)
				throw new NullPointerException();

			String value = element.getAttribute(attribute);
			if (value.length() == 0) {
				if (expectedMissing)
					return defaultValue;
				else
					throw new IllegalArgumentException();
			}

			return Boolean.parseBoolean(value);

		} catch (Exception e) {
			e.printStackTrace();
			System.err.println("Attribute " + attribute + " cannot be read.");
			return defaultValue;
		}
	}

	private int getAttributeValue(Element element, String attribute,
			int defaultValue) {
		try {
			if (element == null)
				throw new NullPointerException();

			String value = element.getAttribute(attribute);
			if (value.length() == 0)
				throw new IllegalArgumentException();

			return Integer.parseInt(value);

		} catch (Exception e) {
			e.printStackTrace();
			System.err.println("Attribute " + attribute + " cannot be read.");
			return defaultValue;
		}
	}

	private Element getLogElement() {
		if (document == null)
			return null;

		NodeList nodes = document.getElementsByTagName("log");
		if (nodes.getLength() < 1) {
			return null;
		} else {
			return (Element) nodes.item(0);
		}
	}

	private Element getEmElement(String purpose) {
		if (document == null)
			return null;

		NodeList nodes = document.getElementsByTagName("em");
		for (int i = 0; i < nodes.getLength(); i++) {
			Element element = (Element) nodes.item(i);
			if (purpose.equals(element.getAttribute("purpose"))) {
				return element;
			}
		}

		return null;
	}

	private EmFramework createEm(Element element,
			CovarianceConstrainer constrainer) {
		if (element == null)
			return null;

		String name = element.getAttribute("name");
		EmFramework em = null;

		if ("LocalEm".equals(name)) {
			em =
					new LocalEm(data, getReuse(element), getRestart(element),
							getSecondStageSteps(element), getThreshold(element));
		} else if ("FullEm".equals(name)) {
			em =
					new FullEm(data, getReuse(element), getRestart(element),
							getMaxSteps(element), getThreshold(element));
		} else
			return null;

		em.useCovarianceConstrainer(constrainer);

		return em;
	}

	private boolean getReuse(Element element) {
		return Boolean.parseBoolean(element.getAttribute("reuse"));
	}

	private int getRestart(Element element) {
		return Integer.parseInt(element.getAttribute("restarts"));
	}

	private int getMaxSteps(Element element) {
		return Integer.parseInt(element.getAttribute("maxSteps"));
	}

	private int getSecondStageSteps(Element element) {
		return Integer.parseInt(element.getAttribute("secondStageSteps"));
	}

	private double getThreshold(Element element) {
		return Double.parseDouble(element.getAttribute("threshold"));
	}

	public CovarianceConstrainer createCovarianceConstrainer() {
		if (document == null)
			return new ConstantCovarianceConstrainer();

		NodeList nodes = document.getElementsByTagName("covarianceConstraints");
		if (nodes.getLength() < 1) {
			return new ConstantCovarianceConstrainer();
		}

		Element element = (Element) nodes.item(0);

		String type = element.getAttribute("type");
		if ("constant".equals(type)) {
			double lower =
					getAttributeValue(element, "eigenvalueLower",
							ConstantCovarianceConstrainer.DEFAULT_LOWER_BOUND,
							true);
			double upper =
					getAttributeValue(element, "eigenvalueUpper",
							ConstantCovarianceConstrainer.DEFAULT_UPPER_BOUND,
							true);
			return new ConstantCovarianceConstrainer(lower, upper);
		} else if ("variable".equals(type)) {
			double multiplier =
					getAttributeValue(element, "multiplier",
							VariableCovarianceConstrainer.DEFAULT_MULTIPLIER,
							true);
			boolean hasUpperBound =
					getAttributeValue(element, "hasUpperBound", true, true);
			return new VariableCovarianceConstrainer(data, multiplier,
					hasUpperBound);
		} else {
			return new ConstantCovarianceConstrainer();
		}
	}
}
