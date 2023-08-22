#' Rank representation over the course of gradient descent
#'
#' Returns the rank representation given in Lemma S1.4.
#'
#' @param j Integer between 1 and n. Index of the item for which the rank is
#' returned.
#' @param t Time over gradient descent. Default is `Inf` which corresponds to
#' the end of training.
#' @param n Number of items. Default is 7.
#' @param alpha Interaction factor (between 0 and 1).
#'
#' @return The rank of the j-th item.
#' @export
gd_rank <- function(j, t=Inf, n=7, alpha=0.) {
  outp <-
    purrr::map(
      seq(1, n, 2),
      function(k) cos((j-1/2)*k*pi/n)*cos(k*pi/(2*n))*(1-exp(-(1-(1-alpha)*cos(k*pi/n))*t))/(1-(1-alpha)*cos(k*pi/n))
    ) %>%
    purrr::reduce(`+`)
  outp <- 2*(1-alpha)/n*outp
  outp
}

#' Adjacent margins over the course of gradient descent
#'
#' Returns the margin for adjacent items over the course of gradient descent.
#'
#' @param j Integer between 1 and n-1.
#' @param t Time over gradient descent. Default is `Inf` which corresponds to
#' the end of training.
#' @param n Number of items. Default is 7.
#' @param alpha Interaction factor (between 0 and 1).
#'
#' @return The margin between item j and j+1.
gd_adjacent_margin <- function(j, t=Inf, n=7, alpha=0.) {
  outp <-
    purrr::map(
      2*(1:as.integer((n+1)/2))-1,
      function(k) sin(j*k*pi/n)*exp(-(1-(1-alpha)*cos(k*pi/n))*t)/tan(k*pi/(2*n))
    ) %>%
    purrr::reduce(`+`)
  outp <- 1-2/n*outp
  outp
}

#' Margin over the course of gradient descent.
#'
#' Returns the margin given in Lemma S1.4.
#'
#' @param j,k Integers between 1 and n.
#' @param t Time over gradient descent. Default is `Inf` which corresponds to
#' the end of training.
#' @param n Number of items. Default is 7.
#' @param alpha Interaction factor (between 0 and 1).
#'
#' @return The margin between item j and item k.
#' @export
gd_margin <- function(j, k, t=Inf, n=7, alpha=0.) {
  dplyr::if_else(
    abs(k-j)==1,
    sign(k-j)*gd_adjacent_margin(pmin(j,k), t=t, n=n, alpha=alpha),
    gd_rank(j, t=t, n=n, alpha=alpha)-gd_rank(k, t=t, n=n, alpha=alpha)
  )
}

#' Average test margin
#'
#' Returns the average test margin
#'
#' @param n Number of items
#' @param alpha Interaction factor
#'
#' @export
average_test_margin <- function(alpha, n=7) {
  if(alpha == 1)
    return(0)
  expand.grid(
    j=1:n,
    k=1:n
  ) %>%
    tibble::as_tibble() %>%
    dplyr::filter(abs(k-j)>1) %>%
    dplyr::mutate(
      margin = abs(gd_margin(j, k, n=n, alpha=alpha))
    ) %>%
    magrittr::extract2('margin') %>%
    mean()
}

#' Margin for regularized regression
#'
#' Returns the margin given in Lemma S1.8.
#'
#' @param j,k Integers between 1 and n.
#' @param c Inverse regularization coefficient Default is `Inf` which corresponds to
#' infinitesimal regularization.
#' @param n Number of items. Default is 7.
#' @param alpha Conjunctivity factor (between 0 and 1).
#'
#' @return The margin between item j and item k.
#' @export
reg_margin <- function(j, k, n=7, alpha=0., c=Inf, lambda=NULL) {
  if(is.null(lambda)){
    lambda <- acosh((1+1/c)/(1-alpha))
  }
  normalization <- sinh((n+1)/2*lambda)-sinh((n-1)/2*lambda)
  difference <- sinh((k-(n+1)/2)*lambda)-sinh((j-(n+1)/2)*lambda)
  m <- (alpha)/(1/c+alpha)
  margin <- difference/normalization
  return(
    dplyr::case_when(
      c == 0 ~ 0.,
      lambda == 0 ~ as.double(k-j),
      abs(j-k) == 1 ~ sign(k-j)*m+(1-m)*margin,
      TRUE ~ margin
    )
  )
}
