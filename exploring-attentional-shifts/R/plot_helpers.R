## Shared plotting helpers for the attention shift trajectory/heatmap plots.
## Every ggplot in analyse.R uses the same iteration-axis breaks, bottom
## legend layout, and ggsave settings; factored out here so each plot only
## has to state what makes it different (data, facets, labels).

#' @title integer_breaks
#' @description Axis break helper that snaps the iteration axis to whole numbers.
integer_breaks <- function(x) {
    seq(ceiling(min(x)), floor(max(x)), by = 1)
}

#' @title bottom_legend_theme
#' @description Shared legend styling (bottom, vertical box, wide keys) used across all trajectory plots.
bottom_legend_theme <- function() {
    theme(
        legend.position = "bottom",
        legend.box = "vertical",
        legend.direction = "horizontal",
        legend.key.width = grid::unit(2.2, "cm"),
        legend.spacing.y = grid::unit(0.3, "cm")
    )
}

#' @title single_row_linetype_guide
#' @description Forces the linetype legend onto a single row beneath the color legend.
single_row_linetype_guide <- function() {
    guides(linetype = guide_legend(order = 2, nrow = 1, byrow = TRUE))
}

#' @title save_figure
#' @description Save a plot into the project's figures directory with the standard export settings.
save_figure <- function(plot, filename, dir, width = 22, height = 11, dpi = 300) {
    if (!dir.exists(dir)) {
        dir.create(dir, recursive = TRUE)
    }
    ggsave(filename = file.path(dir, filename), plot = plot, dpi = dpi, width = width, height = height)
}
