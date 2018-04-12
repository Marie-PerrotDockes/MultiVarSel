#' This function allows the user to select the most relevant variables thanks to
#' the estimation of their selection frequencies obtained by the stability
#' selection approach.
#' @importFrom stats lm ARMAacf ARMAtoMA acf arima model.matrix pchisq
#' @import glmnet
#' @import parallel
#' @import tidyverse
#' @import Matrix
#' @param  X a matrix of covariables
#' @param  Y a response matrix
#' @param  group if the model is an ANOVA, the group resulting from the categorical variable.
#' @param  nb_replis numerical, number of replications in the stability selection
#' @param  nb.cores  numerical, number of cores used
#' @param  typeDep character in c("AR1", "ARMA", "nonparam") defining which type of dependence to use
#' @param  pAR numerical, only use if typeDep = "ARMA", the parameters p for the ARMA(p, q) process
#' @param  qMA numerical, only use if typeDep = "ARMA", the parameters q for the ARMA(p, q) process
#' @param  parallel logical, if TRUE then a parallelized version of the code is used
#' @return A data frame containing the selection frequencies of the different variables obtained
#' by the stability selection, the corresponding level in the design matrix and the associated
#' column of the observations matrix.
#' @examples
#' data("copals_camera")
#' Y <- scale(Y[,1:50])
#' Frequencies <- variable_selection(Y = Y, group = group,
#'  nb_repli = 10, typeDep = 'AR1', pAR = 1, qMA = 0, nb.cores = 4, parallel = FALSE)
#' @export
variable_selection <- function(X, group = NULL, Y, nb_replis = 1000,
                              nb.cores = 4, typeDep = "AR1", pAR = 1, qMA = 0, parallel = FALSE){
  if (!is.null(group)) X <- model.matrix(~ group + 0)
  p <- ncol(X)
  n <- nrow(X)
  q <- ncol(Y)
  nb_repli <- max(nb_replis)

  if (!is.null(group)) {
  sample <- rep(0, length(group))
  s <- sapply(unique(group), function(grp){
    sample[group == grp] <<- c(sample(rep(1:10, sum(group == grp) %/% 10),
                                          10 * (sum(group == grp) %/% 10)),
                               sample(1:10, sum(group == grp) %% 10))})
  }else{
   sample <- c(sample(rep(1:10, n %/% 10),
                          10 * (n %/% 10)),
              sample(1:10, n %% 10))
  }
  Par_fold <- function(i){
    residuals <- lm(as.matrix(Y[sample != i, ]) ~ X[sample != i, ] - 1)$residuals
    square_root_inv_hat_Sigma <- whitening(residuals, typeDep, pAR = pAR, qMA = qMA)
    Yr <- as.numeric(Y[sample != i, ] %*% square_root_inv_hat_Sigma)
    Xr <- kronecker(Matrix::t(square_root_inv_hat_Sigma), X[sample != i, ])
    Lmin <- cv.glmnet(Xr, Yr, intercept = F, parallel = parallel)$lambda.min
    ni <- sum(sample != i)

    Res <- mclapply(1:nb_repli, function(lala){
      if (!is.null(group)) {
      grps <- rep(group[sample != i], q)
      sel  <- sort(unlist(sapply(unique(grps), function(grp){
        sample(which(grps == grp), round(sum(grps == grp) / 2))})))
      }else{
        nb_sel <- ni / 2
        sel <- sample(1:ni, nb_sel) + ni * rep(0:(q - 1), each = nb_sel)
      }
      resultat_glmnet <- glmnet(Xr[sel, ], Yr[sel], family = "gaussian",
                                alpha = 1, lambda = Lmin, intercept = F)
      ind_glmnet  <- which(as.logical(resultat_glmnet$beta != 0))
      return(tabulate(ind_glmnet, (p * q)))
    },
    mc.cores = nb.cores)
    freq <- do.call(cbind, lapply(nb_replis, function(x){
      Reduce("+", Res[1:x]) / x
    }))
    colnames(freq) <- nb_replis
    return(freq)

  }
  if (is.null(colnames(Y))) {
    colnames(Y) <- 1:ncol(Y)
  }
  if (is.null(colnames(X))) {
    colnames(X) <- 1:ncol(X)
  }
  Freqs <- Reduce("+", lapply(1:10, Par_fold)) / 10
  Freqs <- cbind(rep(colnames(Y), each = p), rep(colnames(X),
                                                 q), as.data.frame(Freqs))
  names(Freqs) <- c("Names_of_Y", "Names_of_X", "Frequencies")
  return(Freqs)
}
