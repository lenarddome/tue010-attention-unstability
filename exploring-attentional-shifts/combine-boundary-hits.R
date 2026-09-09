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
    rds_path <- file.path(
        figures_dir,
        paste0("boundary_hits_", condition_name, ".rds")
    )
    if (!file.exists(rds_path)) {
        stop(
            "Missing ", rds_path, " -- run the simulation-*.R script for condition '",
            condition_name, "' first"
        )
    }
    readRDS(rds_path)
})

## The two approaches are not scored out of the same denominator -- the shared
## vector holds one attention value per feature, the attention matrix one per
## outcome row x feature -- so each says what its share is of, right in the
## legend rather than only in the caption. Kept short: these labels sit on one
## line under the panels, so their width is width the panels do not get.
shared_label <- "Shared vector (per feature)"
matrix_label <- "Attention matrix (per outcome row x feature)"

boundary_hits <- rbindlist(lapply(names(conditions), function(condition_label) {
    sweep <- sweeps_by_condition[[condition_label]]
    rbind(
        data.table(sweep$shared, approach = shared_label),
        data.table(sweep$matrix, approach = matrix_label)
    )[, stimulus_set := condition_label]
}))
boundary_hits[, stimulus_set := factor(
    stimulus_set,
    levels = names(conditions)
)]
boundary_hits[, approach := factor(
    approach,
    levels = c(shared_label, matrix_label)
)]

## rho = 0 is the no-shift control: attention never moves, so its curve is a
## flat 0 in every panel of every stimulus set. It is left out of the plot, as
## it is in the per-simulation figures, but the ramp below is still built over
## the whole sweep including 0 -- the sweep each script ran travels with its
## saved tables for exactly this reason. Letting the scale rescale to the
## plotted range instead would make the same ink mean rho = 0.22 here and
## rho = 0 there.
boundary_hits <- boundary_hits[rho > 0]
rho_values <- sort(unique(unlist(
    lapply(sweeps_by_condition, `[[`, "rho_values")
)))

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

## The sweep is a handful of evenly spaced rho values, not a continuum, and a
## colourbar cannot be read back to one of them -- "which curve is rho = 1.33"
## is exactly the question the panels invite. So rho is drawn as an ordered
## discrete scale with one labelled key per simulated value.
##
## The palette is viridis sampled at those same points, which is bit for bit
## what scale_colour_viridis_c(limits = range(rho_values)) assigned before: the
## values are evenly spaced across [0, 2], so sampling n of them reproduces the
## continuous ramp exactly. The floored-at-0 invariant above survives -- rho = 0
## still consumes the darkest end even though no curve is drawn for it.
rho_palette <- setNames(
    viridisLite::viridis(length(rho_values), option = "D"),
    format(round(rho_values, 2), nsmall = 2)
)
boundary_hits[, rho_level := factor(
    format(round(rho, 2), nsmall = 2),
    levels = names(rho_palette)
)]
plotted_rho <- levels(droplevels(boundary_hits$rho_level))

boundary_hits_figure <- ggplot(
    boundary_hits,
    aes(
        x = cycle,
        y = boundary_hit_share,
        colour = rho_level,
        group = draw_group
    )
) +
    geom_line(aes(linetype = approach), linewidth = 0.4) +
    geom_point(aes(shape = approach), size = 2.5) +
    facet_grid(stimulus_set ~ stimulus_name) +
    scale_x_continuous(breaks = integer_breaks) +
    scale_y_continuous(
        limits = c(0, 1),
        breaks = c(0, 0.5, 1),
        labels = scales::percent
    ) +
    scale_colour_manual(values = rho_palette, limits = plotted_rho) +
    scale_linetype_manual(values = c("solid", "22")) +
    scale_shape_manual(values = c(19, 7)) +
    guides(
        ## The keys run high-to-low with no gap between them, so the column still
        ## reads as the ordered ramp the colourbar gave -- but every band now
        ## carries the rho that produced it. Square swatches rather than the
        ## default line-and-glyph key: the line style and the glyph are the other
        ## legend's job, and repeating them here would say twice over something
        ## colour is not encoding.
        colour = guide_legend(
            order = 1,
            position = "right",
            ncol = 1,
            reverse = TRUE,
            override.aes = list(shape = 15, size = 8, linetype = 0),
            theme = theme(
                legend.title.position = "top",
                legend.title = element_text(hjust = 0.5),
                legend.key.width = unit(0.9, "cm"),
                legend.key.height = unit(0.45, "cm"),
                legend.key.spacing.y = unit(0, "pt")
            )
        ),
        ## same title (none) and same guide spec on both scales, so ggplot folds
        ## them into one legend whose keys carry the line style and the glyph
        ## together rather than splitting one variable across two legends.
        ## Under the panels, on one row: it reads along the shared x axis the two
        ## curve families are compared over, and costs no panel width.
        linetype = guide_legend(
            order = 2,
            position = "bottom",
            override.aes = list(colour = "grey25", linewidth = 0.9),
            theme = theme(legend.key.width = unit(1.6, "cm"))
        ),
        shape = guide_legend(order = 2, position = "bottom")
    ) +
    theme_par() +
    labs(
        title = paste(
            "Attention Shifts Pinned at the Lower Boundary Across the Three",
            "Stimulus Sets"
        ),
        x = "Iteration",
        y = "Proportion of Features Reset to 0",
        colour = expression(rho),
        linetype = NULL,
        shape = NULL
    ) +
    theme(
        ## top-justified so the rho column starts level with the panel grid rather
        ## than floating against it
        legend.justification.right = "top",
        legend.box.spacing = unit(0.4, "cm"),
        strip.text.y = element_text(size = 12, angle = -90),
        plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm")
    )

save_figure(
    boundary_hits_figure,
    "figure-2-boundary.png",
    dir = figures_dir,
    width = 12,
    height = 8.5
)
