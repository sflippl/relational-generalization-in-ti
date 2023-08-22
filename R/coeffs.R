k <- function(x, rho) {
  x+rho*(sqrt(1-x**2)-acos(x)*x)/pi
}
k_prime <- function(x, rho) {
  1-rho/pi*acos(x)
}

#' Interaction factor of deep networks
#'
#' Returns the interaction factors of deep networks.
#'
#' @param rho Parameter of nonlinearity (between 0 and 2). `rho=1` corresponds
#' to ReLU
#' @param l Maximal depth
#' @param lambd Proportion of additive nodes. Default is 0.
#' @export
get_kappas <- function(rho, l, lambd=0.) {
  kappa_o <- rep(1/2, length(rho))
  kappa_d <- rep(0, length(rho))
  kappa_o_lst <- list(kappa_o)
  kappa_d_lst <- list(kappa_d)
  for(j in 2:(l+1)){
    kappa_d <- k(kappa_d, rho)
    kappa_o <- (1-lambd)*k(kappa_o, rho)+lambd/2*(1+kappa_d)
    kappa_o_lst <- c(kappa_o_lst, list(kappa_o))
    kappa_d_lst <- c(kappa_d_lst, list(kappa_d))
  }
  list(kappa_o_lst, kappa_d_lst)
}

#' Neural tangent kernel for deep networks
#'
#' Returns the interaction factors of deep networks' NTK.
#'
#' @param rho Parameter of nonlinearity (between 0 and 2). `rho=1` corresponds
#' to ReLU
#' @param l Maximal depth
#' @param lambd Proportion of additive nodes. Default is 0.
#' @export
get_kappas_ntk <- function(rho, l, lambd=0.) {
  kappas <- get_kappas(rho, l, lambd=lambd)
  kappa_o <- rep(1/2, length(rho))
  kappa_d <- rep(0, length(rho))
  kappa_o_lst <- list(kappa_o)
  kappa_d_lst <- list(kappa_d)
  for(j in 2:(l+1)){
    kappa_o <- (kappas[[1]][[j]]+(j-1)*kappa_o*k_prime(kappas[[1]][[j-1]], rho))/j
    kappa_d <- (kappas[[2]][[j]]+(j-1)*kappa_d*k_prime(kappas[[2]][[j-1]], rho))/j
    kappa_o <- (1-lambd)*kappa_o+lambd/2*(1+kappa_d)
    kappa_o_lst <- c(kappa_o_lst, list(kappa_o))
    kappa_d_lst <- c(kappa_d_lst, list(kappa_d))
  }
  list(kappa_o_lst, kappa_d_lst)
}

#' Interaction factors for deep networks
#'
#' Returns interaction factors of deep networks.
#'
#' @param rho Parameter of nonlinearity (between 0 and 2). `rho=1` corresponds
#' to ReLU
#' @param l Maximal depth
#' @param lambd Proportion of additive nodes. Default is 0.
#' @param type If `hidden_layer`, computes the interaction factor of the hidden
#' layers, if `ntk`, computes the interaction factor of the NTK.
#' @export
get_alphas <- function(rho, l, lambd=0., type='hidden_layer') {
  if(type == 'hidden_layer') {
    kappas <- get_kappas(rho, l, lambd=lambd)
  }
  else {
    kappas <- get_kappas_ntk(rho, l, lambd=lambd)
  }
  alphas <- map2(kappas[[1]], kappas[[2]], ~(1+..2-2*..1)/(1-..2))
  alphas
}
