#' This function helps to choose the best whitening strategy among the following types of dependence
#' modellings: AR1, ARMA, non parametric and without any whitening.
#'
#' @param  residuals the residuals matrix of independant linear model on all the collumn of the response matrix
#' @param  typeDeps character in c("AR1", "ARMA", "nonparam", "no_whitening",) defining which estimation of the
#' covariable to use to whitened the residuals.
#' @param  pAR numerical, only use if typeDep = "ARMA", the parameters p for the ARMA(p, q) process
#' @param qMA numerical, only use if typeDep = "ARMA", the parameters q for the ARMA(p, q) process
#' @return It provides a table given the p-values for the different whitening tests apply to residuals multiplied
#'  by the inverse of the square root of the covariance matrix estimated.
#' If the p-value is small (frequently lower than 0.05)
#' it means that the hypothesis that each row of the  residuals "whitened" matrix is a white noise is rejected.
#' @examples
#' data(copals_camera)
#' Y=scale(Y[,1:100])
#' X <- model.matrix( ~ group + 0)
#' residuals=lm(as.matrix(Y)~X-1)$residuals
#' whitening_choice(residuals,c("AR1","nonparam","ARMA"),
#' pAR=1,qMA=1)
#' @export
whitening_choice <- function (residuals, typeDeps = "AR1", pAR = 1, qMA = 0)
{
  get_pvalue <- function(typeDep) {
    square_root_inv_hat_Sigma = whitening(residuals, typeDep,
                                          pAR = pAR, qMA = qMA)
    whitened_residuals = residuals %*% square_root_inv_hat_Sigma
    pvalue = whitening_test(whitened_residuals)
    return(pvalue)
  }
  Pvals <- sapply(typeDeps, get_pvalue)
  Decision <- ifelse(Pvals < 0.05, "NO WHITE NOISE", "WHITE NOISE")
  names(Pvals)[which(names(Pvals) == "ARMA")] <- paste("ARMA",
                                                       pAR, qMA, sep = " ")
  Result <- as.data.frame(cbind(Pvalue = round(Pvals, 3), Decision))
  return(Result)
}
