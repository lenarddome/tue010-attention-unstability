packages <- c("Rcpp", "RcppArmadillo", "data.table", "ggplot2", "pracma", "ggthemes")
lapply(packages, library, character.only = TRUE)
sourceCpp("exploring-attentional-shifts/attention.cpp")

#' @title attention_shift_loop
#' @description Attention shift repeats for a configurable number of iterations, updating the weights and predictions based on the attention shift mechanism.
#' @param weights A matrix of weights representing the connections between nodes.
#' @param predictions A vector of predictions made by the model.
#' @param input A vector representing the input to the model.
#' @param error A vector representing the error between the teacher and predictions.
#' @param gain A scalar representing the gain factor for the attention shift.
#' @param pnorm A scalar representing the p-norm used for normalization in the attention shift.
#' @param P A scalar representing the power used for normalization in the attention shift.
#' @param rho A scalar representing the learning rate for the attention shift.
#' @param n_iterations Number of update iterations to run.
#' @param reduced A boolean indicating whether to use the reduced attention mechanism.
attention_shift_loop <- function(weights, predictions, input, attention, P, rho, teacher, n_iterations = 10, reduced = FALSE) {
    ## calculate input matrix
    input_matrix <- matrix(1, nrow = 5, ncol = 5)
    diag(input_matrix) <- 0 # Create a diagonal matrix with the input vector
    input_matrix <- sweep(input_matrix, 2, t(as.matrix(input)), "*") # Scale the input matrix by the input vector

    ## declare intermediate variables
    attention_prime <- attention * input # Initial attention scaled by input

    ## declare output data tables
    attention_output <- data.table()
    attention_delta_output <- data.table()
    predictions_output <- data.table()
    error_output <- data.table()

    ## loop for n_iterations iterations
    for (i in seq_len(n_iterations)) {
        ## calculate normalised attention gain
        attention_gain <- Norm(attention_prime * input, P)
        if (attention_gain == 0) {
            ## raise error
            message <- paste(
                "Attention gain is zero at iteration", i,
                "with attention_prime:", paste(attention_prime, collapse = ", ")
            )
            stop(message)
        }
        normalised <- (attention_prime * input) / attention_gain
        normalised_matrix <- sweep(input_matrix, 2, t(as.matrix(normalised)), "*") # Scale the input matrix by the normalised attention
        ## calculate model predictions based on current weights, attention and input
        predictions <- rowSums(weights * normalised_matrix * input_matrix) # Simulate predictions as the sum of weights scaled by attention and input
        if (reduced) {
            predictions[1:3] <- 0 # Only consider predictions for active input nodes
        }
        error <- teacher - predictions

        attention_shift <- AttentionShift(weights, predictions, input, error, attention_prime, attention_gain, P, rho)

        attention_prime <- attention_prime + attention_shift * input
        ## clamp attention to be between an arbitraryly small value close to 0 and Infinity
        attention_prime <- pmax(attention_prime, 0.0001)
        ## update weights based on attention shift
        attention_output <- rbind(attention_output, attention_prime)
        predictions_output <- rbind(predictions_output, t(data.matrix(predictions)))
        error_output <- rbind(error_output, t(data.matrix(error)))
        attention_delta_output <- rbind(attention_delta_output, data.matrix(attention_shift))
    }

    return(list(attention = attention_output, predictions = predictions_output, error = error_output, attention_delta = attention_delta_output))
}

set.seed(12)
attention_initial <- runif(5, 0, 1)

set.seed(12)
dist <- rnorm(25, 0.25, 0.5)
dist <- pmax(pmin(dist, 1), -1) # Ensure values are between 0 and 1
weights <- matrix(dist, nrow = 5, ncol = 5)
diag(weights) <- 0 # Set diagonal to 0 to avoid self-connections
print("Initial Weights:")
print(weights)

input <- c(1, 0, 1, 1, 0)
input_matrix <- matrix(1, nrow = 5, ncol = 5)
diag(input_matrix) <- 0 # Create a diagonal matrix with the input vector
input_matrix <- sweep(input_matrix, 2, t(as.matrix(input)), "*") # Scale the input matrix by the input vector
print("Input Matrix:")
print(input_matrix)

teacher <- data.matrix(c(1, 0, 1, 1, 0))
print("Teacher:")
print(teacher)


## make sure weights are somewhat predictive of the teacher to ensure attention shift has an effect
## rerun 5 times to get closer to the teacher
for (i in seq(10)) {
    weights <- sweep(weights, 1, (teacher - rowSums(weights * input_matrix)) * 0.1, "+")
}
diag(weights) <- 0 # Set diagonal to 0 to avoid self-connections
print("Adjusted Weights:")
print(weights)
predictions <- rowSums(weights * input_matrix) # Simulate predictions as the sum of weights scaled by attention and input
error <- teacher - predictions
print("Error:")
print(error)


## create plots for the descent
teacher_labels <- c(
    "1 excitation" = "Single positive value in teacher",
    "2 excitations" = "Two positive values in teacher",
    "3 excitations" = "Three positive values in teacher"
)
## redo everything with different P and rho values to see how it affects the attention shift
## let's do 100 iterations for a more stable result
gradients_grid <- data.table()
attention_values_grid <- data.table()

p_values <- c(5)
rho_values <- seq(0, 2, length.out = 10)
parameter_grid <- CJ(P = p_values, rho = rho_values)

## calculate attention shift loop for different teachers across P/rho combinations
for (param_idx in seq_len(nrow(parameter_grid))) {
    current_P <- parameter_grid[param_idx, P]
    current_rho <- parameter_grid[param_idx, rho]
    print(paste("Running P =", current_P, "and rho =", current_rho))

    for (num_excitations in 1:3) {
        current_teacher <- teacher
        ## flip some of the 1s to 0s to create different teacher patterns
        results <- attention_shift_loop(
            weights,
            predictions,
            input,
            attention_initial,
            P = current_P,
            rho = current_rho,
            teacher = data.matrix(current_teacher),
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
    scale_x_continuous(breaks = function(x) seq(ceiling(min(x)), floor(max(x)), by = 1)) +
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
    guides(
        linetype = guide_legend(order = 2, nrow = 1, byrow = TRUE)
    ) +
    theme(
        legend.position = "bottom",
        legend.box = "vertical",
        legend.direction = "horizontal",
        legend.key.width = grid::unit(2.2, "cm"),
        legend.spacing.y = grid::unit(0.3, "cm"),
        legend.title.position = "left",
    )

ggsave(
    filename = "full_attention_shift_gradients_parameter_grid.png",
    plot = gradient_plot_grid,
    dpi = 300,
    width = 22,
    height = 11
)


## plot attention values across iterations for different P/rho combinations
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
    scale_x_continuous(breaks = function(x) seq(ceiling(min(x)), floor(max(x)), by = 1)) +
    scale_color_viridis_b(option = "D") +
    labs(
        title = "Attention Values for Different Step Size (Rho)",
        x = "Iteration",
        y = "Attention Value",
        color = "Rho",
        linetype = "Input Node"
    ) +
    guides(
        linetype = guide_legend(order = 2, nrow = 1, byrow = TRUE)
    ) +
    theme(
        legend.position = "bottom",
        legend.box = "vertical",
        legend.direction = "horizontal",
        legend.key.width = grid::unit(2.2, "cm"),
        legend.spacing.y = grid::unit(0.3, "cm")
    )

ggsave(
    filename = "full_attention_values_parameter_grid.png",
    plot = attention_values_plot_grid,
    dpi = 300,
    width = 22,
    height = 11
)

## run a similar simulation with teacher and input being different
input <- c(1, 0, 1, 0, 0)
teacher <- c(0, 0, 0, 1, 0)

## repeat the same process as above to see how the attention shift mechanism handles this case where the teacher is not directly aligned with the input
reduced_gradients_grid <- data.table()
reduced_attention_values_grid <- data.table()

reduced_p_values <- c(5)
reduced_rho_values <- seq(0, 2, length.out = 10)
reduced_parameter_grid <- CJ(P = reduced_p_values, rho = reduced_rho_values)

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
    scale_x_continuous(breaks = function(x) seq(ceiling(min(x)), floor(max(x)), by = 1)) +
    theme_par() +
    scale_color_viridis_b(option = "D") +
    labs(
        title = "Reduced Case: Attention Shift Gradients for Different Step Size (Rho)",
        x = "Iteration",
        y = "Attention Shift Gradient",
        color = "Rho",
        linetype = "Input Node"
    ) +
    guides(
        linetype = guide_legend(order = 2, nrow = 1, byrow = TRUE)
    ) +
    theme(
        legend.position = "bottom",
        legend.box = "vertical",
        legend.direction = "horizontal",
        legend.key.width = grid::unit(2.2, "cm"),
        legend.spacing.y = grid::unit(0.3, "cm")
    )

ggsave(
    filename = "reduced_attention_shift_gradients_parameter_grid.png",
    plot = reduced_gradient_plot,
    dpi = 300,
    width = 22,
    height = 11
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
    scale_x_continuous(breaks = function(x) seq(ceiling(min(x)), floor(max(x)), by = 1)) +
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
    guides(
        linetype = guide_legend(order = 2, nrow = 1, byrow = TRUE)
    ) +
    theme(
        legend.position = "bottom",
        legend.box = "vertical",
        legend.direction = "horizontal",
        legend.key.width = grid::unit(2.2, "cm"),
        legend.spacing.y = grid::unit(0.3, "cm")
    )

ggsave(
    filename = "reduced_attention_values_parameter_grid.png",
    plot = reduced_attention_values_plot,
    dpi = 300,
    width = 22,
    height = 11
)

teacher_normalised <- teacher / sum(teacher)

attention_normalised_all <- data.table()
attention_delta_normalised_all <- data.table()

for (current_rho in rho_values) {
    results_normalised <- attention_shift_loop(
        weights,
        predictions,
        input,
        attention_initial,
        P = current_P,
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
    scale_x_continuous(breaks = function(x) seq(ceiling(min(x)), floor(max(x)), by = 1)) +
    theme_par() +
    labs(
        title = "Normalised Attention Values Across Rho Values",
        x = "Iteration",
        y = "Attention Value",
        color = "Input Node"
    ) +
    theme(legend.position = "bottom")

ggsave(
    filename = "normalised_attention_values.png",
    plot = normalised_attention_plot,
    dpi = 300,
    width = 12,
    height = 6
)

normalised_gradient_plot <- ggplot(
    normalised_attention_delta_long,
    aes(x = iteration, y = attention_shift, color = factor(input_node))
) +
    geom_line() +
    facet_wrap(~rho_label, nrow = 1) +
    scale_x_continuous(breaks = function(x) seq(ceiling(min(x)), floor(max(x)), by = 1)) +
    theme_par() +
    labs(
        title = "Normalised Attention Shift Gradients Across Rho Values",
        x = "Iteration",
        y = "Attention Shift Gradient",
        color = "Input Node"
    ) +
    theme(legend.position = "bottom")

ggsave(
    filename = "normalised_attention_shift_gradients.png",
    plot = normalised_gradient_plot,
    dpi = 300,
    width = 12,
    height = 6
)

## now let us swap the attention vector to an attention matrix, and then apply the attention shift to each row of the weight and attention matrices
attention_shift_loop_matrix <- function(weights, input, attention_matrix, P, rho, teacher, n_iterations = 10) {
    n_nodes <- nrow(weights)

    if (!all(dim(weights) == dim(attention_matrix))) {
        stop("weights and attention_matrix must have the same dimensions")
    }
    if (length(input) != ncol(weights)) {
        stop("input length must match number of columns in weights")
    }
    if (length(teacher) != n_nodes) {
        stop("teacher length must match number of rows in weights")
    }

    attention_prime <- attention_matrix
    attention_history <- data.table()
    attention_delta_history <- data.table()
    prediction_history <- data.table()
    error_history <- data.table()

    for (row_idx in seq_len(n_nodes)) {
        for (i in seq_len(n_iterations)) {
            row_attention <- attention_prime[row_idx, ] * input
            attention_gain <- Norm(row_attention, P)
            if (attention_gain == 0) {
                stop(paste("Attention gain is zero at iteration", i, "for row", row_idx))
            }

            normalised_row <- row_attention / attention_gain
            prediction_row <- sum(weights[row_idx, ] * normalised_row * input)
            error_row <- teacher[row_idx] - prediction_row

            attention_shift_row <- AttentionShift(
                matrix(weights[row_idx, ], nrow = 1),
                as.matrix(prediction_row),
                input,
                as.matrix(error_row),
                row_attention,
                attention_gain,
                P,
                rho
            )

            attention_prime[row_idx, ] <- pmax(attention_prime[row_idx, ] + attention_shift_row * input, 0.0001)

            attention_history <- rbind(
                attention_history,
                data.table(
                    iteration = i,
                    row_id = row_idx,
                    input_node = seq_len(ncol(weights)),
                    attention_value = as.numeric(attention_prime[row_idx, ])
                )
            )

            attention_delta_history <- rbind(
                attention_delta_history,
                data.table(
                    iteration = i,
                    row_id = row_idx,
                    input_node = seq_len(ncol(weights)),
                    attention_shift = as.numeric(attention_shift_row)
                )
            )

            prediction_history <- rbind(
                prediction_history,
                data.table(iteration = i, row_id = row_idx, prediction = prediction_row)
            )

            error_history <- rbind(
                error_history,
                data.table(iteration = i, row_id = row_idx, error = error_row)
            )
        }
    }

    return(list(
        attention = attention_history,
        attention_delta = attention_delta_history,
        attention_matrix_final = attention_prime,
        example_row = attention_shift_row
    ))
}


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
    scale_x_continuous(breaks = function(x) seq(ceiling(min(x)), floor(max(x)), by = 1)) +
    theme_par() +
    labs(
        title = "Matrix Attention Shift by Row Across Rho Values",
        x = "Iteration",
        y = "Attention Shift",
        color = "Input Node",
        linetype = "Row"
    ) +
    guides(
        linetype = guide_legend(order = 2, nrow = 1, byrow = TRUE)
    ) +
    theme(
        legend.position = "bottom",
        legend.box = "vertical",
        legend.direction = "horizontal",
        legend.key.width = grid::unit(2.2, "cm"),
        legend.spacing.y = grid::unit(0.3, "cm")
    )

ggsave(
    filename = "row_by_row_attention_shift_grid.png",
    plot = attention_shift_matrix_plot,
    dpi = 300,
    width = 22,
    height = 12
)

attention_values_grid_plot <- ggplot(
    matrix_attention_all,
    aes(x = iteration, y = attention_value, color = factor(input_node), linetype = factor(row_id))
) +
    geom_line() +
    facet_grid(row_id ~ rho_label) +
    scale_x_continuous(breaks = function(x) seq(ceiling(min(x)), floor(max(x)), by = 1)) +
    theme_par() +
    labs(
        title = "Matrix Attention Values by Row Across Rho Values",
        x = "Iteration",
        y = "Attention Value",
        color = "Input Node",
        linetype = "Row"
    ) +
    guides(
        linetype = guide_legend(order = 2, nrow = 1, byrow = TRUE)
    ) +
    theme(
        legend.position = "bottom",
        legend.box = "vertical",
        legend.direction = "horizontal",
        legend.key.width = grid::unit(2.2, "cm"),
        legend.spacing.y = grid::unit(0.3, "cm")
    )

ggsave(
    filename = "row_by_row_attention_values_grid.png",
    plot = attention_values_grid_plot,
    dpi = 300,
    width = 22,
    height = 12
)


## compare initial vs final attention matrix as side-by-side heatmaps
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

ggsave(
    filename = "row_by_row_attention_matrix_initial_final_heatmaps.png",
    plot = attention_heatmap_plot,
    dpi = 300,
    width = 10,
    height = 5
)

### now redo everything but with a normalised teacher signal
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
    scale_x_continuous(breaks = function(x) seq(ceiling(min(x)), floor(max(x)), by = 1)) +
    theme_par() +
    labs(
        title = "Matrix Attention Shift by Row Across Rho Values (Normalised Teacher)",
        x = "Iteration",
        y = "Attention Shift",
        color = "Input Node",
        linetype = "Row"
    ) +
    guides(
        linetype = guide_legend(order = 2, nrow = 1, byrow = TRUE)
    ) +
    theme(
        legend.position = "bottom",
        legend.box = "vertical",
        legend.direction = "horizontal",
        legend.key.width = grid::unit(2.2, "cm"),
        legend.spacing.y = grid::unit(0.3, "cm")
    )

ggsave(
    filename = "row_by_row_attention_shift_grid_normalised_teacher.png",
    plot = normalised_attention_shift_matrix_plot,
    dpi = 300,
    width = 22,
    height = 12
)

normalised_attention_values_grid_plot <- ggplot(
    normalised_matrix_attention_all,
    aes(x = iteration, y = attention_value, color = factor(input_node), linetype = factor(row_id))
) +
    geom_line() +
    facet_grid(row_id ~ rho_label) +
    scale_x_continuous(breaks = function(x) seq(ceiling(min(x)), floor(max(x)), by = 1)) +
    theme_par() +
    labs(
        title = "Matrix Attention Values by Row Across Rho Values (Normalised Teacher)",
        x = "Iteration",
        y = "Attention Value",
        color = "Input Node",
        linetype = "Row"
    ) +
    guides(
        linetype = guide_legend(order = 2, nrow = 1, byrow = TRUE)
    ) +
    theme(
        legend.position = "bottom",
        legend.box = "vertical",
        legend.direction = "horizontal",
        legend.key.width = grid::unit(2.2, "cm"),
        legend.spacing.y = grid::unit(0.3, "cm")
    )

ggsave(
    filename = "row_by_row_attention_values_grid_normalised_teacher.png",
    plot = normalised_attention_values_grid_plot,
    dpi = 300,
    width = 22,
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

ggsave(
    filename = "row_by_row_attention_matrix_initial_final_heatmaps_normalised_teacher.png",
    plot = normalised_attention_heatmap_plot,
    dpi = 300,
    width = 10,
    height = 5
)
