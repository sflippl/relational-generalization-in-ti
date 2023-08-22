#' Compute rank representability
#'
#' The rank representability is defined as the mean squared residual error of
#' approximating the model behavior by a rank representation.
#'
#' @param dat Dataframe with `j` and `k` columns.
#'
#' @return A scalar error.
#' @export
get_rank_rep <- function(dat){
  dat %>%
    dplyr::filter(abs(k-j)>1) %>%
    dplyr::mutate(j = factor(j), k = factor(k)) %>%
    lm(margin ~ j+k, data = .) %>%
    summary() %>%
    magrittr::extract2('residuals') %>%
    magrittr::raise_to_power(2) %>%
    mean()
}
