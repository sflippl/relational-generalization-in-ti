#' Similarity (Experiment 1)
#'
#' This simulation presents the similarity of 20 samples of slim (1000 hidden units)
#' and wide (10,000 hidden units) neural networks with one hidden layer.
#'
#' Each row contains of one item pair.
#'
#' @format ## ``similarity``
#' \describe{
#'   \item{hdims}{Number of hidden units}
#'   \item{model_seed}{Seed of the network simulation}
#'   \item{similarity}{Similarity}
#'   \item{type}{Type of the item pair (same, overlapping, distinct)}
#' }
#' @family {simulations}
"similarity"
