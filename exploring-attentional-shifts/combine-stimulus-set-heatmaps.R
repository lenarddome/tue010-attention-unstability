## Combine the per-condition stimulus-set heatmaps exported by the three
## simulation-*.R scripts (overlapping-all, non-overlapping,
## overlapping-features) into a single publication-ready figure.
##
## Each simulation script saves its two stimulus-set panels -- the final
## attention allocation heatmap (attention matrix vs shared vector) and the
## input-to-output connection sign heatmap -- as figures/panels_<condition>.rds.
## These describe the stimulus set rather than the rho x iteration sweep, so
## they belong together across conditions rather than one pair per results
## figure; the three results figures now carry only the boundary-hit rows.
##
## Reads the saved ggplot objects (rather than re-running the simulations),
## the same pattern combine-exemplar-tables.R uses for the stimulus tables.
##
## Run from the repository root, after the three simulation-*.R scripts have
## each been run at least once. Produces
## figures/figure-stimulus-set-heatmaps.png.

packages <- c("ggplot2", "patchwork")
for (package in packages) {
    if (!requireNamespace(package, quietly = TRUE)) {
        install.packages(package)
    }
    library(package, character.only = TRUE)
}

project_dir <- "exploring-attentional-shifts"
figures_dir <- file.path(project_dir, "figures")
source(file.path(project_dir, "R", "plot_helpers.R"))

## Columns run from least to most outcome-level competition: distinctive
## singular (each stimulus drives one outcome of its own, sharing only input
## features), then distinctive multi-outcome (disjoint outcome sets, several
## outcomes each), then shared outcome spaces (stimuli competing over the same
## outcomes as well as the same features). The names on the left are the
## figure's labels; the slugs on the right are the condition_name each
## simulation script saves its panels under -- overlapping_features is
## simulation 3, non_overlapping simulation 2, overlapping_all simulation 1.
conditions <- c(
    "Distinctive Singular" = "overlapping_features",
    "Distinctive Multi-Outcome" = "non_overlapping",
    "Shared Outcome Spaces" = "overlapping_all"
)

panels_by_condition <- lapply(conditions, function(condition_name) {
    rds_path <- file.path(figures_dir, paste0("panels_", condition_name, ".rds"))
    if (!file.exists(rds_path)) {
        stop("Missing ", rds_path, " -- run the simulation-*.R script for condition '", condition_name, "' first")
    }
    readRDS(rds_path)
})

## The columns are only comparable if every script snapshotted the same point in
## the sweep, which is an agreement between three separate files and so worth
## checking rather than trusting -- a mismatch here means one script's
## heatmap_rho_label index drifted from the others.
snapshot_rho_labels <- vapply(panels_by_condition, `[[`, character(1), "rho_label")
if (length(unique(snapshot_rho_labels)) > 1) {
    stop(
        "Stimulus-set panels were saved at different rho (",
        paste(sprintf("%s: %s", names(snapshot_rho_labels), snapshot_rho_labels), collapse = "; "),
        ") -- make heatmap_rho_label agree across the simulation scripts and re-run them"
    )
}

## ---- Assemble: one column per stimulus set, allocation over weights --------
## Row 1 is the final attention allocation, row 2 the connection sign, so a
## column reads as "this is the weight matrix the run started from, this is
## where its attention ended up". The condition label goes on the row-1 title
## and doubles as the column heading; row 2 drops its own title so the two
## rows stay vertically aligned and the label isn't repeated -- the collected
## legends ("Attention Weights", "Connection Weights") name the two rows.
##
## The allocation fill is pinned to [0, 1] here. Each script already caps its
## attention values at 1, but the per-script scales otherwise end wherever
## that run's maximum falls; a common limit is what makes the three columns
## comparable and lets patchwork collect the three identical scales into one
## legend instead of drawing three subtly different ones.
## All three scripts take this snapshot at the same rho (rho_values[6] = 1.11);
## columns taken at different sweep points would not be comparable. The check
## above holds them to it, and each column's subtitle still names the sweep
## point rather than leaving the reader to assume one.
allocation_panels <- lapply(names(conditions), function(condition_label) {
    panels <- panels_by_condition[[condition_label]]
    panels$allocation +
        scale_fill_viridis_c(option = "D", limits = c(0, 1)) +
        guides(fill = guide_colourbar(theme = theme(
            legend.title.position = "top",
            legend.key.width = unit(4, "cm"),
            legend.key.height = unit(0.4, "cm")
        ))) +
        labs(
            title = condition_label,
            subtitle = sprintf("%s, iteration %d", panels$rho_label, panels$cycle),
            y = "Output Node"
        )
})

weights_panels <- lapply(names(conditions), function(condition_label) {
    panels_by_condition[[condition_label]]$weights +
        guides(fill = guide_legend(theme = theme(legend.title.position = "top"))) +
        labs(title = NULL, y = "Output Node")
})

## Row 1 carries more than row 2 does at the same height -- two stacked facets
## (attention matrix over shared vector), each with its own strip, plus the
## column title and subtitle -- so it gets the larger share of the height.
##
## All six panels share both axes -- the four input nodes across, the outcome
## rows up -- so both titles are collected into one apiece rather than repeated
## six times: the x title centred under the whole grid, the y title centred down
## its left side. Collection matches on the exact string, which is why both
## families set y = "Output Node" above rather than inheriting the per-panel
## labels they were saved with ("Output Node\n" on the allocation heatmaps, ""
## on the weights ones). The two collected legends sit centred beneath the grid,
## each with its title stacked above its keys -- side by side the titles
## otherwise ran into the neighbouring legend's first tick label.
stimulus_set_heatmaps <- wrap_plots(
    c(allocation_panels, weights_panels),
    ncol = length(conditions)
) +
    plot_layout(guides = "collect", heights = c(1.25, 1), axis_titles = "collect") &
    theme(
        legend.position = "bottom",
        legend.box = "vertical",
        legend.justification = "center",
        legend.box.just = "top",
        legend.title = element_text(hjust = 0.5),
        legend.box.spacing = unit(0.6, "cm"),
        legend.spacing.x = unit(1.5, "cm"),
        plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm")
    )

save_figure(
    stimulus_set_heatmaps,
    "figure-1-heatmaps.png",
    dir = figures_dir,
    width = 12.5,
    height = 12.5
)
