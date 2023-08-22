#' Behavioral data from monkeys in Jensen et al. (2015)
#'
#' Data from Jensen et al. (2015) on monkeys performing transitive inference.
#'
#' @format ## `jensen_data_monkeys`
#' \describe{
#'   \item{session}{The session}
#'   \item{target}{The index of the target (between 1 and 6).}
#'   \item{distractor}{The index of the distractor (between 2 and 7).}
#'   \item{acc}{The proportion of correct decisions.}
#' }
#' @source [https://doi.org/10.1371/journal.pcbi.1004523.s002]
#' @family {behavioral datasets}
"jensen_data_monkeys"

#' Behavioral data from humans in Jensen et al. (2015)
#'
#' Data from Jensen et al. (2015) on humans performing transitive inference.
#' This is organised according to symbolic distance as this is the only data
#' we plot in the figures.
#'
#' @format ## `jensen_data_humans`
#' \describe{
#'   \item{symbolic_distance}{Symbolic distance between the target and the
#'   distractor}
#'   \item{subject}{The subject}
#'   \item{acc}{The proportion of correct decisions.}
#' }
#' @source [https://doi.org/10.1371/journal.pcbi.1004523.s002]
#' @family {behavioral datasets}
"jensen_data_humans"

#' Behavioral data from humans in Ciranka et al. (2022)
#'
#' Data from Ciranka et al. (2022) performing transitive inference.
#' This is organised according to symbolic distance as this is the only data
#' we plot in the figures.
#'
#' @format ## `ciranka_data`
#' \describe{
#'   \item{symbolic_distance}{Symbolic distance between the target and the
#'   distractor}
#'   \item{subject}{The subject}
#'   \item{acc}{The proportion of correct decisions.}
#' }
#' @source [https://doi.org/10.1371/journal.pcbi.1004523.s002]
#' @family {behavioral datasets}
"ciranka_data"

