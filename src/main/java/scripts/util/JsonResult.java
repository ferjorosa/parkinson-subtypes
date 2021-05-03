package scripts.util;

public class JsonResult {

    private double learningTime;
    private double elbo;
    private double logLikelihood;
    private double bic;
    private double aic;
    private int nVbsemCandidates;
    private int nVbemCandidates;

    public JsonResult(double learningTime,
                      double elbo,
                      double logLikelihood,
                      double bic,
                      double aic,
                      int nVbsemCandidates,
                      int nVbemCandidates) {

        this.learningTime = learningTime;
        this.elbo = elbo;
        this.logLikelihood = logLikelihood;
        this.bic = bic;
        this.aic = aic;
        this.nVbsemCandidates = nVbsemCandidates;
        this.nVbemCandidates = nVbemCandidates;
    }
}
