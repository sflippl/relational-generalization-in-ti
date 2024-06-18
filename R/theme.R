#' Standard theme
#'
#' This theme is used for the manuscript's figures.
#' @export
theme_princti <-
  function() {
    ggplot2::theme_classic() +
      ggplot2::theme(
        title = ggplot2::element_text(size = 8),
        text = ggplot2::element_text(size = 7, family='Helvetica'),
        axis.text = ggplot2::element_text(size = 7),
        legend.text = ggplot2::element_text(size = 7),
        plot.tag = ggplot2::element_text(face = 'bold', size = 8),
        strip.text = ggplot2::element_text(size = 7),
        strip.background = element_blank()
      )
  }

#' Void theme
#'
#' This theme is used for the manuscript's figures if they should rely on
#' [ggplot2::theme_void()]
#' @export
theme_princti_void <-
  function() {
    ggplot2::theme_void() +
      ggplot2::theme(
        plot.tag = ggplot2::element_text(face = 'bold', size = 8)
      )
  }
