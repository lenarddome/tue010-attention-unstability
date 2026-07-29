## Multi-outcome attention shift simulation: a comparative study of a single
## shared attention vector versus a per-outcome attention matrix, both
## exposed to several distinct stimulus->outcome pairs in round-robin
## rotation. This directly tests the README's claim that shared attention
## vectors become unstable under multi-outcome learning, by comparing them
## against a condition (the attention matrix) that has no cross-outcome
## sharing at all -- mirroring the shared-vector-vs-attention-matrix
## comparison analyse.R already runs for the single-stimulus case.
##
## Unlike analyse.R (auto-associative: the same 5 nodes serve as both
## stimulus features and outcome nodes, square weights), this uses a
## bipartite model: 4 stimulus-feature nodes are a separate population from
## 5 outcome nodes (rectangular weights), so there is no self-connection to
## suppress (suppress_diagonal = FALSE throughout). Three stimuli share
## overlapping active features so attention faces genuine competition when
## the active stimulus rotates every iteration.
##
## Both conditions reuse attention_shift_loop() and attention_shift_loop_matrix()
## from R/simulate.R unmodified in spirit -- generalized in place to accept
## rectangular weights and an optional suppress_diagonal flag -- rather than
## duplicating their logic. The round-robin rotation itself is driven from
## this script by calling each function once per stimulus with
## n_iterations = 1, carrying the returned attention forward as the next
## call's starting point (see run_round_robin_shared() / _matrix() below).
##
## Run from the repository root. Produces PNGs in exploring-attentional-shifts/figures/.

## ---- Setup ------------------------------------------------------------------
packages <- c("Rcpp", "RcppArmadillo", "data.table", "ggplot2", "pracma", "ggthemes", "patchwork", "gridExtra")
lapply(packages, library, character.only = TRUE)

project_dir <- "exploring-attentional-shifts"
figures_dir <- file.path(project_dir, "figures")

sourceCpp(file.path(project_dir, "attention.cpp"))
source(file.path(project_dir, "R", "simulate.R"))
source(file.path(project_dir, "R", "plot_helpers.R"))

## ---- Round-robin drivers ------------------------------------------------------
## Repeatedly call the existing single-stimulus functions with n_iterations = 1,
## rotating through `stimuli` and carrying the returned attention state forward.
## preserve_inactive_attention = TRUE / no re-masking (attention_shift_loop_matrix
## never re-masks by input) means a feature's attention survives intact across
## iterations where it is not part of the active stimulus.
run_round_robin_shared <- function(weights, attention_initial, stimuli, P, rho, n_iterations) {
    n_stimuli <- length(stimuli)
    attention_current <- attention_initial

    attention_output <- data.table()
    attention_delta_output <- data.table()

    for (i in seq_len(n_iterations)) {
        stimulus_id <- ((i - 1) %% n_stimuli) + 1
        cycle <- ((i - 1) %/% n_stimuli) + 1
        s <- stimuli[[stimulus_id]]

        result <- attention_shift_loop(
            weights,
            predictions = numeric(nrow(weights)),
            input = s$input,
            attention = attention_current,
            P = P,
            rho = rho,
            teacher = s$teacher,
            n_iterations = 10,
            suppress_diagonal = FALSE,
            preserve_inactive_attention = TRUE
        )

        attention_current <- as.numeric(result$attention[1, ])
        attention_shift_current <- as.numeric(result$attention_delta[1, ])

        attention_output <- rbind(attention_output, data.table(
            iteration = i, cycle = cycle, stimulus_id = stimulus_id, stimulus_name = s$name,
            feature_node = seq_along(attention_current), attention_value = attention_current
        ))
        attention_delta_output <- rbind(attention_delta_output, data.table(
            iteration = i, cycle = cycle, stimulus_id = stimulus_id, stimulus_name = s$name,
            feature_node = seq_along(attention_shift_current), attention_shift = attention_shift_current
        ))
    }

    list(attention = attention_output, attention_delta = attention_delta_output)
}

run_round_robin_matrix <- function(weights, attention_matrix_initial, stimuli, P, rho, n_iterations) {
    n_stimuli <- length(stimuli)
    attention_current <- attention_matrix_initial

    attention_output <- data.table()
    attention_delta_output <- data.table()

    for (i in seq_len(n_iterations)) {
        stimulus_id <- ((i - 1) %% n_stimuli) + 1
        cycle <- ((i - 1) %/% n_stimuli) + 1
        s <- stimuli[[stimulus_id]]

        result <- attention_shift_loop_matrix(
            weights,
            input = s$input,
            attention_matrix = attention_current,
            P = P,
            rho = rho,
            teacher = s$teacher,
            n_iterations = 1,
            suppress_diagonal = FALSE
        )

        attention_current <- result$attention_matrix_final

        this_attention <- result$attention[, .(row_id, feature_node = input_node, attention_value)]
        this_attention[, `:=`(iteration = i, cycle = cycle, stimulus_id = stimulus_id, stimulus_name = s$name)]
        attention_output <- rbind(attention_output, this_attention)

        this_delta <- result$attention_delta[, .(row_id, feature_node = input_node, attention_shift)]
        this_delta[, `:=`(iteration = i, cycle = cycle, stimulus_id = stimulus_id, stimulus_name = s$name)]
        attention_delta_output <- rbind(attention_delta_output, this_delta)
    }

    list(attention = attention_output, attention_delta = attention_delta_output)
}

## ---- Stimulus set -------------------------------------------------------------
## 4 feature nodes -> 5 outcome nodes. Active-feature overlap across stimuli is
## deliberately graded to create a spectrum of competition for attention:
##   feature 1 (A & B only): teachers {2,5} vs {1,3} share no outcome  -> pure conflict
##   feature 2 (A & C):      teachers {2,5} vs {3,4,5} share outcome 5 -> partial conflict
##   feature 3 (B & C):      teachers {1,3} vs {3,4,5} share outcome 3 -> partial conflict
##   feature 4 (C only):     no other stimulus touches it              -> no competition (control)
stimuli <- list(
    list(name = "A", input = c(1, 0, 0, 0), teacher = c(0, 1, 0, 0, 1)),
    list(name = "B", input = c(0, 0, 1, 0), teacher = c(1, 0, 1, 0, 0)),
    list(name = "C", input = c(0, 1, 0, 1), teacher = c(0, 0, 1, 1, 1))
)

n_features <- length(stimuli[[1]]$input)
n_outcomes <- length(stimuli[[1]]$teacher)
n_stimuli <- length(stimuli)
stimulus_names <- vapply(stimuli, `[[`, character(1), "name")

## only a stimulus's own active features can ever have their attention shift;
## used to convert boundary-hit counts into a share of the values that were
## actually eligible to be affected, so stimuli with different numbers of
## active features (A/B have 2, C has 3) become directly comparable
n_active_by_stimulus <- data.table(
    stimulus_name = stimulus_names,
    n_active = vapply(stimuli, function(s) sum(s$input), numeric(1))
)

attention_initial <- runif(n_features, 0, 0.1)

## the matrix (per-row) condition starts from the same vector broadcast to
## every row, so any divergence between the two conditions comes from
## whether attention is shared across outcomes, not from a different start
attention_matrix_initial <- matrix(rep(attention_initial, each = n_outcomes), nrow = n_outcomes)

## ---- Weights: pretrain round-robin across all three pairs --------------------
set.seed(12)
dist <- rnorm(n_outcomes * n_features, 0.25, 0.5)
dist <- pmax(pmin(dist, 1), -1) # Clamp to [-1, 1]
weights <- matrix(dist, nrow = n_outcomes, ncol = n_features) # no diag zeroing: rectangular, no self-connections
print("Initial Weights:")
print(weights)

## round-robin the pretraining over all 3 pairs each epoch, rather than
## sequentially finishing one before moving to the next, so weights converge
## to a genuine multi-way compromise instead of overfitting one pair and then
## being dragged away from it. The nudge is gated by which features are
## active for the current stimulus (outer(error, input)) rather than applied
## uniformly across the whole row: an ungated nudge would also perturb
## columns the current stimulus never touches, corrupting the fit for
## whichever other stimulus relies on them.
for (epoch in seq_len(50)) {
    for (s in stimuli) {
        input_matrix_s <- sweep(matrix(1, nrow = n_outcomes, ncol = n_features), 2, s$input, "*")
        predictions_s <- rowSums(weights * input_matrix_s)
        weights <- weights + outer((s$teacher - predictions_s) * 0.1, s$input)
    }
}
print("Adjusted Weights:")
print(weights)

for (s in stimuli) {
    input_matrix_s <- sweep(matrix(1, nrow = n_outcomes, ncol = n_features), 2, s$input, "*")
    predictions_s <- rowSums(weights * input_matrix_s)
    print(paste("Residual error for stimulus", s$name, ":"))
    print(s$teacher - predictions_s)
}

## ---- P/rho parameter grid, round-robin rotation through all three stimuli ----
## Each stimulus gets its own full n_cycles = 10 repetitions, matching the
## n_iterations = 10 convention used throughout analyse.R -- the rotation
## still interleaves stimuli every global step (A, B, C, A, B, C, ...), but
## every stimulus is presented exactly n_cycles times, so per-stimulus views
## (like the boundary-hit heatmaps below) have no missing/inapplicable cells.
p_values <- c(7.5)
rho_values <- seq(0, 2, length.out = 10)
parameter_grid <- CJ(P = p_values, rho = rho_values)

n_cycles <- 10
n_iterations_multi <- n_cycles * n_stimuli

attention_multi_grid <- data.table()
attention_delta_multi_grid <- data.table()
attention_matrix_multi_grid <- data.table()
attention_delta_matrix_multi_grid <- data.table()

for (param_idx in seq_len(nrow(parameter_grid))) {
    current_P <- parameter_grid[param_idx, P]
    current_rho <- parameter_grid[param_idx, rho]
    print(paste("Running P =", current_P, "and rho =", current_rho))

    results_shared <- run_round_robin_shared(weights, attention_initial, stimuli, current_P, current_rho, n_iterations_multi)
    attention_multi_grid <- rbind(
        attention_multi_grid,
        data.table(results_shared$attention, P = current_P, rho = current_rho, grid_id = param_idx)
    )
    attention_delta_multi_grid <- rbind(
        attention_delta_multi_grid,
        data.table(results_shared$attention_delta, P = current_P, rho = current_rho, grid_id = param_idx)
    )

    results_matrix <- run_round_robin_matrix(weights, attention_matrix_initial, stimuli, current_P, current_rho, n_iterations_multi)
    attention_matrix_multi_grid <- rbind(
        attention_matrix_multi_grid,
        data.table(results_matrix$attention, P = current_P, rho = current_rho, grid_id = param_idx)
    )
    attention_delta_matrix_multi_grid <- rbind(
        attention_delta_matrix_multi_grid,
        data.table(results_matrix$attention_delta, P = current_P, rho = current_rho, grid_id = param_idx)
    )
}

attention_multi_grid[, rho_label := sprintf("rho=%.2f", rho)]
attention_delta_multi_grid[, rho_label := sprintf("rho=%.2f", rho)]
attention_matrix_multi_grid[, rho_label := sprintf("rho=%.2f", rho)]
attention_delta_matrix_multi_grid[, rho_label := sprintf("rho=%.2f", rho)]

stimulus_phase <- unique(attention_multi_grid[, .(iteration, stimulus_name, rho_label)])

## ---- Plot (a): shared-vector attention trajectories --------------------------
## One continuous line per feature node across the whole rotation (not
## faceted by stimulus, which would fragment each line and hide the
## cross-switch persistence that is the whole point).
attention_multi_plot <- ggplot() +
    geom_rect(
        data = stimulus_phase,
        aes(xmin = iteration - 0.5, xmax = iteration + 0.5, ymin = -Inf, ymax = Inf, fill = stimulus_name),
        alpha = 0.15,
        inherit.aes = FALSE
    ) +
    geom_line(
        data = attention_multi_grid,
        aes(x = iteration, y = attention_value, color = factor(feature_node))
    ) +
    facet_wrap(~rho_label, nrow = 2) +
    scale_x_continuous(breaks = integer_breaks) +
    scale_fill_manual(values = c("A" = "grey75", "B" = "grey45", "C" = "grey15")) +
    theme_par() +
    labs(
        title = "Shared Attention Trajectories Across a Round-Robin Stimulus Rotation",
        x = "Iteration",
        y = "Attention Value",
        color = "Feature Node",
        fill = "Active Stimulus"
    ) +
    bottom_legend_theme()

save_figure(attention_multi_plot, "overlapping_outcomes_multi_outcome_shared_attention_values.png", dir = figures_dir)

## ---- Plot (b): shared-vector attention shift boundary hits -------------------
## The raw gradient trajectories were too busy to read at a glance (every
## feature node's shift, every iteration, every rho, all overplotted). Instead,
## count how often a shift saturates at the LOWER tanh boundary specifically
## (shift = -1, i.e. AttentionShift's rho * pnorm^-1 term has grown large
## enough to saturate while driving attention toward the clamp floor -- the
## collapse direction the destabilization story is actually about; the upper
## boundary (+1) is not what we care about here), then convert that count
## into a share of the values that were actually eligible to be affected
## (only a stimulus's own active features can shift at all), so stimuli with
## different numbers of active features become directly comparable on one
## 0-100% scale. Shown as a heatmap over stimulus repetition x rho, one panel
## per stimulus. `cycle` (1..n_cycles) is each stimulus's own repetition
## count -- since every stimulus is presented exactly n_cycles times, every
## (cycle, stimulus) cell has a real value and no facet needs padding.
boundary_hits_shared <- attention_delta_multi_grid[
    , .(boundary_hits = sum(attention_shift <= -1 + 1e-9)),
    by = .(cycle, stimulus_name, rho, rho_label)
]
boundary_hits_shared <- merge(boundary_hits_shared, n_active_by_stimulus, by = "stimulus_name")
boundary_hits_shared[, boundary_hit_share := boundary_hits / n_active]

boundary_hits_shared_plot <- ggplot(
    boundary_hits_shared,
    aes(x = factor(cycle, levels = seq_len(n_cycles)), y = rho_label, fill = boundary_hit_share)
) +
    geom_tile(color = "white") +
    facet_wrap(~stimulus_name, nrow = 1) +
    scale_fill_viridis_c(option = "D", labels = scales::percent, limits = c(0, 1)) +
    theme_par() +
    labs(
        title = "Shared Attention Shift: Share of Active Features at Lower Boundary",
        x = "Stimulus Repetition",
        y = "Rho",
        fill = "Share at\nLower Boundary"
    ) +
    theme(axis.text.y = element_text(size = 7))


## ---- Plot (a2): matrix (per-row) attention trajectories, row_by_row style ----
## Mirrors analyse.R's row_by_row_* plots: every outcome row has its own
## private attention vector, so faceting by row_id shows each row's
## trajectory separately.
attention_matrix_multi_plot <- ggplot() +
    geom_rect(
        data = stimulus_phase,
        aes(xmin = iteration - 0.5, xmax = iteration + 0.5, ymin = -Inf, ymax = Inf, fill = stimulus_name),
        alpha = 0.15,
        inherit.aes = FALSE
    ) +
    geom_line(
        data = attention_matrix_multi_grid,
        aes(x = iteration, y = attention_value, color = factor(feature_node))
    ) +
    facet_grid(row_id ~ rho_label) +
    scale_x_continuous(breaks = integer_breaks) +
    scale_fill_manual(values = c("A" = "grey75", "B" = "grey45", "C" = "grey15")) +
    theme_par() +
    labs(
        title = "Per-Row Attention Matrix Trajectories Across a Round-Robin Stimulus Rotation",
        x = "Iteration",
        y = "Attention Value",
        color = "Feature Node",
        fill = "Active Stimulus"
    ) +
    bottom_legend_theme()

save_figure(attention_matrix_multi_plot, "overlapping_outcomes_multi_outcome_matrix_attention_values.png", dir = figures_dir, height = 12)

## ---- Plot (b2): matrix (per-row) attention shift boundary hits ---------------
## Same lower-boundary-hit count as above (shift = -1 only), summed across
## both outcome rows and feature nodes, then converted to a share of the
## eligible row x active-feature cells (n_outcomes * n_active for that
## stimulus) -- the matrix analogue of the shared-vector share plot above, on
## the same 0-100% scale for direct comparison. Again keyed on each
## stimulus's own repetition count (`cycle`) rather than the global iteration.
boundary_hits_matrix <- attention_delta_matrix_multi_grid[
    , .(boundary_hits = sum(attention_shift <= -1 + 1e-9)),
    by = .(cycle, stimulus_name, rho, rho_label)
]
boundary_hits_matrix <- merge(boundary_hits_matrix, n_active_by_stimulus, by = "stimulus_name")
boundary_hits_matrix[, boundary_hit_share := boundary_hits / (n_active * n_outcomes)]

boundary_hits_matrix_plot <- ggplot(
    boundary_hits_matrix,
    aes(x = factor(cycle, levels = seq_len(n_cycles)), y = rho_label, fill = boundary_hit_share)
) +
    geom_tile(color = "white") +
    facet_wrap(~stimulus_name, nrow = 1) +
    scale_fill_viridis_c(option = "D", labels = scales::percent, limits = c(0, 1)) +
    theme_par() +
    labs(
        title = "Attention Matrix Shift: Share of Row x Active-Feature Cells at Lower Boundary",
        x = "Stimulus Repetition",
        y = "Rho",
        fill = "Share at\nLower Boundary"
    ) +
    theme(axis.text.y = element_text(size = 7))


## ---- Comparative instability plot: shared vector vs attention matrix --------
## Direct test of the destabilization claim, comparing the two conditions
## head to head: within-cycle variance of each feature's attention value,
## the first complete cycle vs the last complete cycle. For the matrix
## condition, a feature's attention is averaged across outcome rows first so
## it sits on the same footing as the shared vector's single value per
## feature. Feature 1 (pure conflict between A and B) is expected to show
## the steepest variance growth under the shared condition specifically;
## feature 4 (uncontested, C only) is expected to stay flat under both.
shared_for_comparison <- copy(attention_multi_grid)
shared_for_comparison[, approach := "Shared vector"]

matrix_for_comparison <- attention_matrix_multi_grid[
    ,
    .(attention_value = mean(attention_value)),
    by = .(iteration, cycle, stimulus_id, stimulus_name, feature_node, P, rho, grid_id, rho_label)
]
matrix_for_comparison[, approach := "Attention matrix (row-averaged)"]

comparison_grid <- rbind(shared_for_comparison, matrix_for_comparison, use.names = TRUE)

variance_by_cycle <- comparison_grid[
    ,
    .(attention_var = var(attention_value)),
    by = .(feature_node, rho, P, cycle, approach)
]
variance_by_cycle[, cycle_phase := fifelse(
    cycle == 1, "Early (cycle 1)",
    fifelse(cycle == n_cycles, paste0("Late (cycle ", n_cycles, ")"), "Middle")
)]

instability_plot <- ggplot(
    variance_by_cycle[cycle_phase != "Middle"],
    aes(x = rho, y = attention_var, color = factor(feature_node))
) +
    geom_line() +
    geom_point() +
    facet_grid(cycle_phase ~ approach) +
    theme_par() +
    labs(
        title = "Per-Feature Attention Instability: Shared Vector vs Attention Matrix",
        x = "Rho",
        y = "Within-Cycle Attention Variance",
        color = "Feature Node"
    ) +
    theme(legend.position = "bottom")

save_figure(
    instability_plot,
    "overlapping_outcomes_multi_outcome_instability_shared_vs_matrix.png",
    dir = figures_dir,
    width = 14,
    height = 9
)

## ---- Comparative heatmap: final attention allocation ------------------------
## At the highest rho in the grid, the state after all n_iterations_multi
## iterations. The attention matrix genuinely has row structure (one vector
## per outcome node); the shared vector is a single vector, not a matrix. Each
## gets its own panel (facet_grid with space = "free_y" sizes each panel's
## height by its own number of rows, so the single-row shared-vector panel
## isn't stretched to match the 5-row matrix panel) rather than forcing both
## onto one shared y-axis.
heatmap_rho_label <- sprintf("rho=%.2f", rho_values)[6]
final_iteration <- n_iterations_multi

final_matrix_state <- attention_matrix_multi_grid[
    rho_label == heatmap_rho_label & cycle == 10,
    .(panel = "Attention Matrix", row_label = as.character(row_id), feature_node, attention_value)
]
final_matrix_state <- final_matrix_state[, attention_value := sum(attention_value), by = .(panel, row_label, feature_node)]

final_shared_state <- attention_multi_grid[
    rho_label == heatmap_rho_label & cycle == 10,
    .(panel = "Shared Vector", row_label = "", feature_node, attention_value)
]
final_shared_state <- final_shared_state[, attention_value := sum(attention_value), by = .(panel, row_label, feature_node)]

## normalise each run to sum to 1 -- each outcome row's own vector for the
## matrix condition, and the single shared vector as a whole -- so the
## heatmap shows how attention is *allocated* across features within a run
## rather than raw magnitudes that differ in scale from run to run.
# final_matrix_state[, attention_value := attention_value / sum(attention_value), by = row_label]
# final_shared_state[, attention_value := attention_value / sum(attention_value)]

final_combined <- rbind(final_matrix_state, final_shared_state)
final_combined[, panel := factor(panel, levels = c("Attention Matrix", "Shared Vector"))]

final_combined$attention_value[final_combined$attention_value > 1] <- 1

attention_allocation_heatmap <- ggplot(
    final_combined,
    aes(x = factor(feature_node), y = row_label, fill = attention_value)
) +
    geom_tile(color = "white") +
    facet_grid(panel ~ ., scales = "free_y", space = "free_y") +
    scale_fill_viridis_c(option = "D") +
    theme_par() +
    labs(
        title = "Attention Matrix vs Shared Vector",
        x = "Feature Node",
        y = "Outcome Row",
        fill = "Share of\nRun's Attention"
    )

## ---- Design panel: the stimulus/outcome patterns being compared -------------
## A reference panel to the right of the heatmap showing the experimental
## design itself -- which features are active per stimulus, and which
## outcome each stimulus targets -- so the attention allocation above can be
## read against what each stimulus actually asked for.
design_input_table <- as.data.frame(do.call(rbind, lapply(stimuli, `[[`, "input")))
colnames(design_input_table) <- paste0("F", seq_len(n_features))
rownames(design_input_table) <- stimulus_names

design_output_table <- as.data.frame(do.call(rbind, lapply(stimuli, `[[`, "teacher")))
colnames(design_output_table) <- paste0("O", seq_len(n_outcomes))
rownames(design_output_table) <- stimulus_names

design_input_plot <- wrap_elements(panel = tableGrob(design_input_table, theme = ttheme_default(
    core = list(fg_params = list(hjust = 1, x = 0.9)),
    rowhead = list(fg_params = list(hjust = 1.5, x = 0.95))
))) +
    ggtitle("Input Features")
design_output_plot <- wrap_elements(panel = gridExtra::tableGrob(design_output_table, theme = ttheme_default(
    core = list(fg_params = list(hjust = 1, x = 0.9)),
    rowhead = list(fg_params = list(hjust = 1.5, x = 0.95))
))) +
    ggtitle("Feedback Vectors")

design_panel <- design_input_plot / design_output_plot

## ---- Combine: heatmap+design on top, boundary-hit heatmaps as two rows below ----
attention_allocation_combined <- (attention_allocation_heatmap | design_panel) /
    boundary_hits_shared_plot /
    boundary_hits_matrix_plot +
    plot_layout(widths = c(1, 1.5), heights = c(1.2, 1, 1)) +
    plot_annotation(
        title = "Final Attention Allocation vs Experimental Design",
        subtitle = paste("At", heatmap_rho_label, "after", n_cycles, "repetitions per stimulus")
    ) &
    theme(plot.margin = margin(2, 2, 2, 2))

save_figure(
    attention_allocation_combined,
    "overlapping_outcomes_multi_outcome_attention_allocation_heatmap.png",
    dir = figures_dir,
    width = 10,
    height = 12
)
