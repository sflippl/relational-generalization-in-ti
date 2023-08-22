#' TI performance of neural networks at different scales
#'
#' Predictions of the network at different output scales.
#'
#' @format ## ``ti_scaling``
#' \describe{
#'   \item{scaling}{Constant by which the output was scaled}
#'   \item{model_seed}{Random seed of the model initialization}
#'   \item{j,k}{Indices of the items being compared}
#'   \item{margin} {Network output}
#' }
#'
#' @family {simulations}
#'
"ti_scaling"

#' TI performance's rank representability of neural networks at different scales
#'
#' Predictions of the network at different output scales.
#'
#' @format ## ``ti_scaling_rank_rep``
#' \describe{
#'   \item{scaling}{Constant by which the output was scaled}
#'   \item{model_seed}{Random seed of the model initialization}
#'   \item{rank_rep} {Rank representability}
#' }
#'
#' @family {simulations}
#'
"ti_scaling_rank_rep"
