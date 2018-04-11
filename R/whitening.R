#' This function provides an estimation of the inverse of the square root of the covariance matrix of each row of the residuals matrix.
#'
#' @param  residuals the residuals matrix obtained by fitting a linear model to each column of the response matrix as if they were independent
#' @param  typeDep character in c("AR1", "ARMA", "nonparam") defining which type of dependence to use
#' @param  pAR numerical, only use if typeDep = "ARMA", the parameter p for the ARMA(p, q) process
#' @param  qMA numerical, only use if typeDep = "ARMA", the parameter q for the ARMA(p, q) process
#' @return It returns the estimation of the inverse of the square root of the covariance matrix of each row of the residuals matrix.
#' @examples
#' data(copals_camera)
#' Y=scale(Y[,1:100])
#' X <- model.matrix( ~ group + 0)
#' residuals=lm(as.matrix(Y)~X-1)$residuals
#' whitening(residuals, "AR1")
#' @export
whitening <- function (residuals, typeDep, pAR = 1, qMA = 0){
  n = dim(residuals)[1]
  q = dim(residuals)[2]
  if (typeDep == "no_whitening") {
    return(Diagonal(q, 1))
  }
  if (typeDep == "AR1") {
    phi_hat <- t(apply(residuals, 1, function(x){
      arima(x, order = c(1, 0, 0))$coef[1]
    }))
    phi_hat_final <- mean(phi_hat)
    phi_hat_vect <- rep( - phi_hat_final, (q - 1))
    square_root_inv_hat_Sigma <- bandSparse(q, k = c(1, 0),
                                            diagonals = list(phi_hat_vect,
                                                             c(sqrt(1 - phi_hat_final^2),
                                                               rep(1, (q - 1)))
                                            )
    )
    return(square_root_inv_hat_Sigma)
  }
  if (typeDep == "ARMA") {
    Coefs <- apply(residuals, 1, function(x){
      arima(x, order = c(pAR, 0, qMA))$coef
    })
    if (pAR > 1) { phi_hat_final <-  rowMeans(Coefs[1:pAR, ])
    }else{
      phi_hat_final <-  ifelse(pAR == 0, 0, mean(Coefs[1:pAR, ]))
    }

    if (qMA > 1) { theta_hat_final <-  rowMeans(Coefs[(pAR + 1):(pAR + qMA), ])
    }else{
      theta_hat_final <-  ifelse(qMA == 0, 0, mean(Coefs[(pAR + 1):(pAR + qMA), ]))
    }

    acf_theo_hat <- ARMAacf(ar = phi_hat_final, ma = theta_hat_final,
                            lag.max = (q - 1))
    psi_hat <- ARMAtoMA(ar = phi_hat_final,
                        ma = theta_hat_final, 1000)
    variance_hat <- 1 + sum(psi_hat ^ 2)
    Sigma_hat <- toeplitz(acf_theo_hat) * variance_hat
    square_root_inv_hat_Sigma <- Matrix(round(solve(chol(Sigma_hat)),
                                              digits = 6))
    return(square_root_inv_hat_Sigma)
  }
  if (typeDep == "nonparam") {
    vector_cov <- t(apply(residuals, 1, function(x){
      acf(x, type = "covariance",
          plot = FALSE, lag.max = (q - 1))$acf
    }))

    vector_cov_estim <- colMeans(vector_cov)
    cov_matrix <- toeplitz(vector_cov_estim)
    square_root_inv_hat_Sigma <- Matrix(round(solve(chol(cov_matrix)),
                                              digits = 6))
    return(square_root_inv_hat_Sigma)
  }
}
