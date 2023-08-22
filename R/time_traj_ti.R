#' Time trajectory of deep network training on TI (Exp. 2)
#'
#' Predictions of the network over the duration of gradient descent.
#'
#' @format ## ``time_traj_ti``
#' \describe{
#'   \item{scaling}{Constant by which the output was scaled}
#'   \item{model_seed}{Random seed of the model initialization}
#'   \item{j,k}{Indices of the items being compared}
#'   \item{margin} {Network output}
#'   \item{epoch}{Training epoch}
#' }
#'
#' @family {simulations}
#'
"time_traj_ti"

#' Learning rates of Exp. 2
#'
#' Learning rates in Exp. 2
"time_traj_ti_lr"

#' Time trajectory of TI rank representability (Exp. 2)
#'
#' Rank representability of the network over the duration of gradient descent.
#'
#' @format ## ``time_traj_ti``
#' \describe{
#'   \item{scaling}{Constant by which the output was scaled}
#'   \item{model_seed}{Random seed of the model initialization}
#'   \item{rank_rep} {Rank representability}
#'   \item{epoch}{Training epoch}
#' }
#'
#' @family {simulations}
#'
"time_traj_ti_rank_rep"
