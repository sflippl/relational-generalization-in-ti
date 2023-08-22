#' TI behavior for different piecewise linear activation functions
#'
#' @format ## ``nonlinearity_ti``
#' \describe{
#'   \item{rho}{Scalar parameterizing the activation function (see (59))}
#'   \item{model_seed}{Random seed of the model initialization}
#'   \item{j,k}{Indices of the items being compared}
#'   \item{margin} {Network output}
#' }
#'
#' @family {simulations}
"nonlinearity_ti"

#' Rank representability for different piecewise linear activation functions
#'
#' @format ## ``nonlinearity_ti_rank_rep``
#' \describe{
#'   \item{rho}{Scalar parameterizing the activation function (see (59))}
#'   \item{model_seed}{Random seed of the model initialization}
#'   \item{rank_rep} {Rank representability}
#' }
#'
#' @family {simulations}
"nonlinearity_ti_rank_rep"
