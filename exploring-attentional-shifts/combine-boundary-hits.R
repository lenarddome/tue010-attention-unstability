## Combine the per-condition boundary-hit sweeps exported by the three
## simulation-*.R scripts (overlapping-all, non-overlapping,
## overlapping-features) into a single publication-ready figure.
##
## Each simulation script saves the two tables behind its own results figure --
## the share of eligible attention shifts pinned at the lower boundary, per
## presentation and per rho, for the shared vector and for the attention matrix
## -- as figures/boundary_hits_<condition>.rds. This script puts all three
## stimulus sets in one grid so the central comparison (the shared vector
## saturates, the attention matrix does not, in every stimulus set) can be read
## without flipping between figures.
##
## Reads the saved tables rather than re-running the simulations, the same
## pattern combine-exemplar-tables.R and combine-stimulus-set-heatmaps.R use.
##
## Run from the repository root, after the three simulation-*.R scripts have
## each been run at least once. Produces figures/figure-boundary-hits.png.

packages <- c("ggplot2", "data.table", "ggthemes")
for (package in packages) {
    if (!requireNamespace(package, quietly = TRUE)) {
        install.packages(package)
    }
    library(package, character.only = TRUE)
}

project_dir <- "exploring-attentional-shifts"
figures_dir <- file.path(project_dir, "figures")
source(file.path(project_dir, "R", "plot_helpers.R"))

## Same row order and display names as combine-stimulus-set-heatmaps.R: least to
## most outcome-level competition. The slugs on the right are the condition_name
## each simulation script saves under -- overlapping_features is simulation 3,
## non_overlapping simulation 2, overlapping_all simulation 1.
conditions <- c(
    "Distinctive Singular" = "overlapping_features",
    "Distinctive Multi-Outcome" = "non_overlapping",
    "Shared Outcome Spaces" = "overlapping_all"
)

sweeps_by_condition <- lapply(conditions, function(condition_name) {
    rds_path <- file.path(figures_dir, paste0("boundary_hits_", condition_name, ".rds"))
    if (!file.exists(rds_path)) {
        stop("Missing ", rds_path, " -- run the simulation-*.R script for condition '", condition_name, "' first")
    }
    readRDS(rds_path)
})

## The two approaches are not scored out of the same denominator -- the shared
## vector holds one attention value per feature, the attention matrix one per
## outcome row x feature -- so each says what its share is of, right in the
## legend rather than only in the caption.
shared_label <- "Shared vector\n(share of active features)"
matrix_label <- "Attention matrix\n(share of outcome row x feature cells)"

boundary_hits <- rbindlist(lapply(names(conditions), function(condition_label) {
    sweep <- sweeps_by_condition[[condition_label]]
    rbind(
        data.table(sweep$shared, approach = shared_label),
        data.table(sweep$matrix, approach = matrix_label)
    )[, stimulus_set := condition_label]
}))
boundary_hits[, stimulus_set := factor(stimulus_set, levels = names(conditions))]
boundary_hits[, approach := factor(approach, levels = c(shared_label, matrix_label))]

## rho = 0 is the no-shift control: attention never moves, so its curve is a flat
## 0 in every panel of every stimulus set. It is left out of the plot, as it is
## in the per-simulation figures, but the ramp below is still floored at 0 --
## the sweep each script ran travels with its saved tables for exactly this
## reason. Letting the scale rescale to the plotted range instead would make the
## same ink mean rho = 0.22 here and rho = 0 there.
boundary_hits <- boundary_hits[rho > 0]
rho_limits <- range(unlist(lapply(sweeps_by_condition, `[[`, "rho_values")))

## Both conditions carry the rho ramp; solid + circle vs dashed + triangle tells
## them apart. Colour is doing one job here (rho, the ordered sweep parameter)
## and line style plus glyph the other (which attention representation), so
## neither has to double up, and the matrix condition keeps its own rho
## structure instead of collapsing into a single contextual grey. The two
## conditions cross and overlap inside a shared panel, which is why the approach
## is worth encoding twice over.
##
## Descending rho order for the same reason as in the simulation scripts: where
## curves coincide, the visible one is the lowest rho that reaches that path.
## ggplot draws one group at a time in level order, so the draw order has to be
## carried by an explicit factor built in the sorted data's own row order --
## matrix first so the shared curves, which carry the claim, sit on top of it.
setorder(boundary_hits, -approach, -rho)
boundary_hits[, draw_group := factor(
    paste(approach, rho, sep = " | "),
    levels = unique(paste(approach, rho, sep = " | "))
)]

boundary_hits_figure <- ggplot(
    boundary_hits,
    aes(x = cycle, y = boundary_hit_share, colour = rho, group = draw_group)
) +
    geom_line(aes(linetype = approach), linewidth = 0.4) +
    geom_point(aes(shape = approach), size = 2.5) +
    facet_grid(stimulus_set ~ stimulus_name) +
    scale_x_continuous(breaks = integer_breaks) +
    scale_y_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1), labels = scales::percent) +
    scale_colour_viridis_c(option = "D", limits = rho_limits) +
    scale_linetype_manual(values = c("solid", "22")) +
    scale_shape_manual(values = c(19, 7)) +
    guides(
        colour = guide_colourbar(order = 1, theme = theme(
            legend.title.position = "top",
            legend.key.width = unit(4, "cm"),
            legend.key.height = unit(0.4, "cm")
        )),
        ## same title (none) and same guide spec on both scales, so ggplot folds
        ## them into one legend whose keys carry the line style and the glyph
        ## together rather than splitting one variable across two legends
        linetype = guide_legend(
            order = 2,
            override.aes = list(colour = "grey25", linewidth = 0.9)
        ),
        shape = guide_legend(order = 2)
    ) +
    theme_par() +
    labs(
        title = "Attention Shifts Pinned at the Lower Boundary Across the Three Stimulus Sets",
        x = "Iteration",
        y = "Proportion of Stimuli Reset to 0",
        colour = expression(rho),
        linetype = NULL,
        shape = NULL
    ) +
    theme(
        legend.position = "bottom",
        legend.box = "vertical",
        legend.justification = "center",
        legend.box.just = "top",
        legend.title = element_text(hjust = 0.5),
        legend.box.spacing = unit(0.6, "cm"),
        legend.spacing.x = unit(1.5, "cm"),
        strip.text.y = element_text(size = 12, angle = -90),
        plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm")
    )

save_figure(
    boundary_hits_figure,
    "figure-boundary-hits.png",
    dir = figures_dir,
    width = 13,
    height = 10
)
