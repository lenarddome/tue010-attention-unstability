## Attention shift simulations: how a shared attention vector behaves under
## multi-outcome learning, compared against a per-row attention matrix.
##
## Run from the repository root. Produces PNGs in exploring-attentional-shifts/figures/.

## ---- Setup ------------------------------------------------------------------
packages <- c("Rcpp", "RcppArmadillo", "data.table", "ggplot2", "pracma", "ggthemes")
lapply(packages, library, character.only = TRUE)

project_dir <- "exploring-attentional-shifts"
figures_dir <- file.path(project_dir, "figures")

sourceCpp(file.path(project_dir, "attention.cpp"))
source(file.path(project_dir, "R", "simulate.R"))
source(file.path(project_dir, "R", "plot_helpers.R"))

## ---- Initial weights, input, and teacher signal ------------------------------
set.seed(12)
attention_initial <- runif(5, 0, 1)

set.seed(12)
dist <- rnorm(25, 0.25, 0.5)
dist <- pmax(pmin(dist, 1), -1) # Clamp to [-1, 1]
weights <- matrix(dist, nrow = 5, ncol = 5)
diag(weights) <- 0 # No self-connections
print("Initial Weights:")
print(weights)

input <- c(1, 0, 1, 1, 0)
input_matrix <- matrix(1, nrow = 5, ncol = 5)
diag(input_matrix) <- 0
input_matrix <- sweep(input_matrix, 2, t(as.matrix(input)), "*")
print("Input Matrix:")
print(input_matrix)

teacher <- data.matrix(c(1, 0, 1, 1, 0))
print("Teacher:")
print(teacher)

## nudge weights toward the teacher so the attention shift has a visible effect
for (i in seq(10)) {
    weights <- sweep(weights, 1, (teacher - rowSums(weights * input_matrix)) * 0.1, "+")
}
diag(weights) <- 0
print("Adjusted Weights:")
print(weights)

predictions <- rowSums(weights * input_matrix)
error <- teacher - predictions
print("Error:")
print(error)

## ---- Vector attention model: P/rho parameter grid ----------------------------
teacher_labels <- c(
    "1 excitation" = "Single positive value in teacher",
    "2 excitations" = "Two positive values in teacher",
    "3 excitations" = "Three positive values in teacher"
)

p_values <- c(5)
rho_values <- seq(0, 2, length.out = 10)
parameter_grid <- CJ(P = p_values, rho = rho_values)

gradients_grid <- data.table()
attention_values_grid <- data.table()

for (param_idx in seq_len(nrow(parameter_grid))) {
    current_P <- parameter_grid[param_idx, P]
    current_rho <- parameter_grid[param_idx, rho]
    print(paste("Running P =", current_P, "and rho =", current_rho))

    for (num_excitations in 1:3) {
        results <- attention_shift_loop(
            weights,
            predictions,
            input,
            attention_initial,
            P = current_P,
            rho = current_rho,
            teacher = data.matrix(teacher),
            reduced = FALSE
        )

        gradients_grid <- rbind(
            gradients_grid,
            data.table(
                results$attention_delta,
                teacher = num_excitations,
                iteration = seq_len(10),
                P = current_P,
                rho = current_rho,
                grid_id = param_idx
            )
        )
        attention_values_grid <- rbind(
            attention_values_grid,
            data.table(
                results$attention,
                teacher = num_excitations,
                iteration = seq_len(10),
                P = current_P,
                rho = current_rho,
                grid_id = param_idx
            )
        )
    }
}

gradients_grid_long <- melt(
    gradients_grid,
    id.vars = c("teacher", "iteration", "P", "rho", "grid_id"),
    variable.name = "input_node",
    value.name = "attention_shift"
)
gradients_grid_long[, teacher_name := factor(
    paste(teacher, ifelse(teacher == 1, "excitation", "excitations")),
    levels = names(teacher_labels),
    labels = unname(teacher_labels)
)]

gradient_plot_grid <- ggplot(
    gradients_grid_long,
    aes(x = iteration, y = attention_shift, linetype = input_node, color = rho)
) +
    geom_line() +
    scale_x_continuous(breaks = integer_breaks) +
    facet_grid(teacher_name ~ grid_id) +
    theme_par() +
    scale_color_viridis_b(option = "D") +
    labs(
        title = "Attention Shift Gradients for Different Step Size (Rho)",
        x = "Iteration",
        y = "Attention Shift Gradient",
        color = "Rho",
        linetype = "Input Node"
    ) +
    single_row_linetype_guide() +
    bottom_legend_theme() +
    theme(legend.title.position = "left")

save_figure(
    gradient_plot_grid,
    "full_attention_shift_gradients_parameter_grid.png",
    dir = figures_dir
)

attention_values_grid_long <- melt(
    attention_values_grid,
    id.vars = c("teacher", "iteration", "P", "rho", "grid_id"),
    variable.name = "input_node",
    value.name = "attention_value"
)
attention_values_grid_long[, teacher_name := factor(
    paste(teacher, ifelse(teacher == 1, "excitation", "excitations")),
    levels = names(teacher_labels),
    labels = unname(teacher_labels)
)]

attention_values_plot_grid <- ggplot(
    attention_values_grid_long,
    aes(x = iteration, y = attention_value, linetype = input_node, color = rho)
) +
    geom_line() +
    facet_grid(teacher_name ~ grid_id) +
    theme_par() +
    scale_x_continuous(breaks = integer_breaks) +
    scale_color_viridis_b(option = "D") +
    labs(
        title = "Attention Values for Different Step Size (Rho)",
        x = "Iteration",
        y = "Attention Value",
        color = "Rho",
        linetype = "Input Node"
    ) +
    single_row_linetype_guide() +
    bottom_legend_theme()

save_figure(
    attention_values_plot_grid,
    "full_attention_values_parameter_grid.png",
    dir = figures_dir
)

## ---- Vector attention model: reduced case (misaligned teacher) ---------------
## Same process as above, but the teacher is not directly aligned with the input.
input <- c(1, 0, 1, 0, 0)
teacher <- c(0, 0, 0, 1, 0)

reduced_p_values <- c(5)
reduced_rho_values <- seq(0, 2, length.out = 10)
reduced_parameter_grid <- CJ(P = reduced_p_values, rho = reduced_rho_values)

reduced_gradients_grid <- data.table()
reduced_attention_values_grid <- data.table()

for (param_idx in seq_len(nrow(reduced_parameter_grid))) {
    current_P <- reduced_parameter_grid[param_idx, P]
    current_rho <- reduced_parameter_grid[param_idx, rho]
    print(paste("Reduced case: Running P =", current_P, "and rho =", current_rho))

    results <- attention_shift_loop(
        weights,
        predictions,
        input,
        attention_initial,
        P = current_P,
        rho = current_rho,
        teacher = data.matrix(teacher),
        n_iterations = 10,
        reduced = TRUE
    )

    reduced_gradients_grid <- rbind(
        reduced_gradients_grid,
        data.table(
            results$attention_delta,
            iteration = seq_len(10),
            P = current_P,
            rho = current_rho,
            grid_id = param_idx
        )
    )
    reduced_attention_values_grid <- rbind(
        reduced_attention_values_grid,
        data.table(
            results$attention,
            iteration = seq_len(10),
            P = current_P,
            rho = current_rho,
            grid_id = param_idx
        )
    )
}

reduced_gradients_long <- melt(
    reduced_gradients_grid,
    id.vars = c("iteration", "P", "rho", "grid_id"),
    variable.name = "input_node",
    value.name = "attention_shift"
)

reduced_gradient_plot <- ggplot(
    reduced_gradients_long,
    aes(x = iteration, y = attention_shift, linetype = input_node, color = rho)
) +
    geom_line() +
    facet_wrap(~grid_id, nrow = 1) +
    scale_x_continuous(breaks = integer_breaks) +
    theme_par() +
    scale_color_viridis_b(option = "D") +
    labs(
        title = "Reduced Case: Attention Shift Gradients for Different Step Size (Rho)",
        x = "Iteration",
        y = "Attention Shift Gradient",
        color = "Rho",
        linetype = "Input Node"
    ) +
    single_row_linetype_guide() +
    bottom_legend_theme()

save_figure(
    reduced_gradient_plot,
    "reduced_attention_shift_gradients_parameter_grid.png",
    dir = figures_dir
)

reduced_attention_values_long <- melt(
    reduced_attention_values_grid,
    id.vars = c("iteration", "P", "rho", "grid_id"),
    variable.name = "input_node",
    value.name = "attention_value"
)

reduced_attention_values_plot <- ggplot(
    reduced_attention_values_long,
    aes(x = iteration, y = attention_value, linetype = input_node, color = rho)
) +
    geom_line() +
    scale_x_continuous(breaks = integer_breaks) +
    facet_wrap(~grid_id, nrow = 1) +
    theme_par() +
    scale_color_viridis_b(option = "D") +
    labs(
        title = "Reduced Case: Attention Values for Different Step Size (Rho)",
        x = "Iteration",
        y = "Attention Value",
        color = "Rho",
        linetype = "Input Node"
    ) +
    single_row_linetype_guide() +
    bottom_legend_theme()

save_figure(
    reduced_attention_values_plot,
    "reduced_attention_values_parameter_grid.png",
    dir = figures_dir
)

## ---- Vector attention model: normalised teacher -----------------------------
teacher_normalised <- teacher / sum(teacher)

attention_normalised_all <- data.table()
attention_delta_normalised_all <- data.table()

for (current_rho in rho_values) {
    results_normalised <- attention_shift_loop(
        weights,
        predictions,
        input,
        attention_initial,
        P = 5,
        rho = current_rho,
        teacher = data.matrix(teacher_normalised),
        reduced = FALSE
    )

    attention_normalised_all <- rbind(
        attention_normalised_all,
        data.table(results_normalised$attention, rho = current_rho, iteration = seq_len(10))
    )

    attention_delta_normalised_all <- rbind(
        attention_delta_normalised_all,
        data.table(results_normalised$attention_delta, rho = current_rho, iteration = seq_len(10))
    )
}

normalised_attention_long <- melt(
    attention_normalised_all,
    id.vars = c("rho", "iteration"),
    variable.name = "input_node",
    value.name = "attention_value"
)

normalised_attention_delta_long <- melt(
    attention_delta_normalised_all,
    id.vars = c("rho", "iteration"),
    variable.name = "input_node",
    value.name = "attention_shift"
)

normalised_attention_long[, rho_label := sprintf("rho=%.2f", rho)]
normalised_attention_delta_long[, rho_label := sprintf("rho=%.2f", rho)]

normalised_attention_plot <- ggplot(
    normalised_attention_long,
    aes(x = iteration, y = attention_value, color = factor(input_node))
) +
    geom_line() +
    facet_wrap(~rho_label, nrow = 1) +
    scale_x_continuous(breaks = integer_breaks) +
    theme_par() +
    labs(
        title = "Normalised Attention Values Across Rho Values",
        x = "Iteration",
        y = "Attention Value",
        color = "Input Node"
    ) +
    theme(legend.position = "bottom")

save_figure(
    normalised_attention_plot,
    "normalised_attention_values.png",
    dir = figures_dir,
    width = 12,
    height = 6
)

normalised_gradient_plot <- ggplot(
    normalised_attention_delta_long,
    aes(x = iteration, y = attention_shift, color = factor(input_node))
) +
    geom_line() +
    facet_wrap(~rho_label, nrow = 1) +
    scale_x_continuous(breaks = integer_breaks) +
    theme_par() +
    labs(
        title = "Normalised Attention Shift Gradients Across Rho Values",
        x = "Iteration",
        y = "Attention Shift Gradient",
        color = "Input Node"
    ) +
    theme(legend.position = "bottom")

save_figure(
    normalised_gradient_plot,
    "normalised_attention_shift_gradients.png",
    dir = figures_dir,
    width = 12,
    height = 6
)

## ---- Matrix attention model: P/rho parameter grid ----------------------------
## Swap the shared attention vector for a per-row attention matrix, applying
## the attention shift to each row of the weight/attention matrices in turn.
attention_matrix_initial <- matrix(rep(attention_initial, each = nrow(weights)), nrow = nrow(weights))
diag(attention_matrix_initial) <- 0
teacher <- data.matrix(c(1, 0, 1, 1, 0))
input <- c(1, 0, 1, 1, 0)

matrix_rho_values <- seq(0, 2, length.out = 10)
matrix_attention_delta_all <- data.table()
matrix_attention_all <- data.table()

for (current_rho in matrix_rho_values) {
    results_matrix <- attention_shift_loop_matrix(
        weights = weights,
        input = input,
        attention_matrix = attention_matrix_initial,
        P = 5,
        rho = current_rho,
        teacher = as.vector(teacher),
        n_iterations = 10
    )

    matrix_attention_delta_all <- rbind(
        matrix_attention_delta_all,
        data.table(results_matrix$attention_delta, rho = current_rho)
    )

    matrix_attention_all <- rbind(
        matrix_attention_all,
        data.table(results_matrix$attention, rho = current_rho)
    )
}

matrix_attention_delta_all[, rho_label := sprintf("rho=%.2f", rho)]
matrix_attention_all[, rho_label := sprintf("rho=%.2f", rho)]

attention_shift_matrix_plot <- ggplot(
    matrix_attention_delta_all,
    aes(x = iteration, y = attention_shift, color = factor(input_node), linetype = factor(row_id))
) +
    geom_line() +
    facet_grid(row_id ~ rho_label) +
    scale_x_continuous(breaks = integer_breaks) +
    theme_par() +
    labs(
        title = "Matrix Attention Shift by Row Across Rho Values",
        x = "Iteration",
        y = "Attention Shift",
        color = "Input Node",
        linetype = "Row"
    ) +
    single_row_linetype_guide() +
    bottom_legend_theme()

save_figure(
    attention_shift_matrix_plot,
    "row_by_row_attention_shift_grid.png",
    dir = figures_dir,
    height = 12
)

attention_values_grid_plot <- ggplot(
    matrix_attention_all,
    aes(x = iteration, y = attention_value, color = factor(input_node), linetype = factor(row_id))
) +
    geom_line() +
    facet_grid(row_id ~ rho_label) +
    scale_x_continuous(breaks = integer_breaks) +
    theme_par() +
    labs(
        title = "Matrix Attention Values by Row Across Rho Values",
        x = "Iteration",
        y = "Attention Value",
        color = "Input Node",
        linetype = "Row"
    ) +
    single_row_linetype_guide() +
    bottom_legend_theme()

save_figure(
    attention_values_grid_plot,
    "row_by_row_attention_values_grid.png",
    dir = figures_dir,
    height = 12
)

## ---- Matrix attention model: initial vs final heatmap ------------------------
heatmap_rho <- tail(matrix_rho_values, 1)
results_matrix_heatmap <- attention_shift_loop_matrix(
    weights = weights,
    input = input,
    attention_matrix = attention_matrix_initial,
    P = 5,
    rho = heatmap_rho,
    teacher = as.vector(teacher),
    n_iterations = 10
)

initial_attention_long <- melt(
    as.data.table(attention_matrix_initial)[, row_id := .I],
    id.vars = "row_id",
    variable.name = "input_node",
    value.name = "attention_value"
)
initial_attention_long[, input_node := as.integer(gsub("V", "", input_node))]
initial_attention_long[, stage := "Initial"]

final_attention_long <- melt(
    as.data.table(results_matrix_heatmap$attention_matrix_final)[, row_id := .I],
    id.vars = "row_id",
    variable.name = "input_node",
    value.name = "attention_value"
)
final_attention_long[, input_node := as.integer(gsub("V", "", input_node))]
final_attention_long[, stage := "Final"]

attention_heatmap_long <- rbind(initial_attention_long, final_attention_long)
attention_heatmap_long[, is_diagonal := row_id == input_node]
attention_heatmap_long[is_diagonal == TRUE, attention_value := NA_real_]

attention_heatmap_plot <- ggplot(
    attention_heatmap_long,
    aes(x = input_node, y = row_id, fill = attention_value)
) +
    geom_tile(color = "white") +
    geom_text(
        data = attention_heatmap_long[is_diagonal == TRUE],
        aes(label = "x"),
        color = "black",
        size = 4
    ) +
    facet_wrap(~stage, nrow = 1) +
    coord_fixed() +
    scale_fill_viridis_c(option = "D", na.value = "white") +
    theme_par() +
    scale_x_continuous(breaks = seq(1, ncol(weights), by = 1)) +
    scale_y_continuous(breaks = seq(1, nrow(weights), by = 1)) +
    labs(
        title = "Initial vs Final Attention Matrix",
        subtitle = paste("Final matrix after", 10, "iterations with rho =", heatmap_rho),
        x = "Input Node (Column)",
        y = "Weight Row",
        fill = "Attention"
    )

save_figure(
    attention_heatmap_plot,
    "row_by_row_attention_matrix_initial_final_heatmaps.png",
    dir = figures_dir,
    width = 10,
    height = 5
)

## ---- Matrix attention model: normalised teacher -----------------------------
teacher_vector <- as.vector(teacher)
if (sum(teacher_vector) == 0) {
    stop("Cannot normalise teacher signal because its sum is zero")
}
teacher_normalised <- teacher_vector / sum(teacher_vector)

normalised_matrix_attention_delta_all <- data.table()
normalised_matrix_attention_all <- data.table()

for (current_rho in rho_values) {
    results_matrix_normalised <- attention_shift_loop_matrix(
        weights = weights,
        input = input,
        attention_matrix = attention_matrix_initial,
        P = 5,
        rho = current_rho,
        teacher = teacher_normalised,
        n_iterations = 10
    )

    normalised_matrix_attention_delta_all <- rbind(
        normalised_matrix_attention_delta_all,
        data.table(results_matrix_normalised$attention_delta, rho = current_rho)
    )

    normalised_matrix_attention_all <- rbind(
        normalised_matrix_attention_all,
        data.table(results_matrix_normalised$attention, rho = current_rho)
    )
}

normalised_matrix_attention_delta_all[, rho_label := sprintf("rho=%.2f", rho)]
normalised_matrix_attention_all[, rho_label := sprintf("rho=%.2f", rho)]

normalised_attention_shift_matrix_plot <- ggplot(
    normalised_matrix_attention_delta_all,
    aes(x = iteration, y = attention_shift, color = factor(input_node), linetype = factor(row_id))
) +
    geom_line() +
    facet_grid(row_id ~ rho_label) +
    scale_x_continuous(breaks = integer_breaks) +
    theme_par() +
    labs(
        title = "Matrix Attention Shift by Row Across Rho Values (Normalised Teacher)",
        x = "Iteration",
        y = "Attention Shift",
        color = "Input Node",
        linetype = "Row"
    ) +
    single_row_linetype_guide() +
    bottom_legend_theme()

save_figure(
    normalised_attention_shift_matrix_plot,
    "row_by_row_attention_shift_grid_normalised_teacher.png",
    dir = figures_dir,
    height = 12
)

normalised_attention_values_grid_plot <- ggplot(
    normalised_matrix_attention_all,
    aes(x = iteration, y = attention_value, color = factor(input_node), linetype = factor(row_id))
) +
    geom_line() +
    facet_grid(row_id ~ rho_label) +
    scale_x_continuous(breaks = integer_breaks) +
    theme_par() +
    labs(
        title = "Matrix Attention Values by Row Across Rho Values (Normalised Teacher)",
        x = "Iteration",
        y = "Attention Value",
        color = "Input Node",
        linetype = "Row"
    ) +
    single_row_linetype_guide() +
    bottom_legend_theme()

save_figure(
    normalised_attention_values_grid_plot,
    "row_by_row_attention_values_grid_normalised_teacher.png",
    dir = figures_dir,
    height = 12
)

heatmap_rho_normalised <- tail(rho_values, 1)
results_matrix_heatmap_normalised <- attention_shift_loop_matrix(
    weights = weights,
    input = input,
    attention_matrix = attention_matrix_initial,
    P = 5,
    rho = heatmap_rho_normalised,
    teacher = teacher_normalised,
    n_iterations = 10
)

normalised_initial_attention_long <- melt(
    as.data.table(attention_matrix_initial)[, row_id := .I],
    id.vars = "row_id",
    variable.name = "input_node",
    value.name = "attention_value"
)
normalised_initial_attention_long[, input_node := as.integer(gsub("V", "", input_node))]
normalised_initial_attention_long[, stage := "Initial"]

normalised_final_attention_long <- melt(
    as.data.table(results_matrix_heatmap_normalised$attention_matrix_final)[, row_id := .I],
    id.vars = "row_id",
    variable.name = "input_node",
    value.name = "attention_value"
)
normalised_final_attention_long[, input_node := as.integer(gsub("V", "", input_node))]
normalised_final_attention_long[, stage := "Final"]

normalised_attention_heatmap_long <- rbind(normalised_initial_attention_long, normalised_final_attention_long)
normalised_attention_heatmap_long[, is_diagonal := row_id == input_node]
normalised_attention_heatmap_long[is_diagonal == TRUE, attention_value := NA_real_]

normalised_attention_heatmap_plot <- ggplot(
    normalised_attention_heatmap_long,
    aes(x = input_node, y = row_id, fill = attention_value)
) +
    geom_tile(color = "white") +
    geom_text(
        data = normalised_attention_heatmap_long[is_diagonal == TRUE],
        aes(label = "x"),
        color = "black",
        size = 4
    ) +
    facet_wrap(~stage, nrow = 1) +
    coord_fixed() +
    scale_fill_viridis_c(option = "D", na.value = "white") +
    theme_par() +
    scale_x_continuous(breaks = seq(1, ncol(weights), by = 1)) +
    scale_y_continuous(breaks = seq(1, nrow(weights), by = 1)) +
    labs(
        title = "Initial vs Final Attention Matrix (Normalised Teacher)",
        subtitle = paste("Final matrix after", 10, "iterations with rho =", heatmap_rho_normalised, "and normalised teacher signal"),
        x = "Input Node (Column)",
        y = "Weight Row",
        fill = "Attention"
    )

save_figure(
    normalised_attention_heatmap_plot,
    "row_by_row_attention_matrix_initial_final_heatmaps_normalised_teacher.png",
    dir = figures_dir,
    width = 10,
    height = 5
)
