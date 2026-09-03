## Core simulation functions for the attention shift model.
## Requires AttentionShift() and Norm() (from attention.cpp / pracma) to be
## available in the calling environment.

#' @title attention_shift_loop
#' @description Attention shift repeats for a configurable number of iterations, updating the weights and predictions based on the attention shift mechanism.
#' @param weights A matrix of weights representing the connections between nodes, n_outcomes rows by n_features columns (square for the auto-associative model; rectangular for a bipartite stimulus/outcome model).
#' @param predictions A vector of predictions made by the model.
#' @param input A vector representing the input to the model.
#' @param attention A vector representing the initial attention weights.
#' @param P A scalar representing the power used for normalization in the attention shift.
#' @param rho A scalar representing the learning rate for the attention shift.
#' @param teacher A vector representing the target output of the model.
#' @param n_iterations Number of update iterations to run.
#' @param reduced A boolean indicating whether to use the reduced attention mechanism.
#' @param suppress_diagonal Passed through to AttentionShift(); also gates the R-side diagonal zeroing below. Default TRUE preserves the original auto-associative behavior (self-connections suppressed). Pass FALSE for a bipartite model where stimulus features and outcome nodes are separate populations and there is no self-connection to suppress.
#' @param preserve_inactive_attention If FALSE (default), attention is reset to `attention * input` before the loop starts, zeroing any node not active in `input` -- correct when `input` is fixed for the whole loop (the original single-stimulus use). If TRUE, the passed-in `attention` is used as-is, leaving values for currently-inactive nodes untouched; needed when chaining repeated single-iteration calls across a rotation of different stimuli, so that attention accrued while a feature was active persists once that feature goes quiet.
attention_shift_loop <- function(weights, predictions, input, attention, P, rho, teacher, n_iterations = 10,
                                  reduced = FALSE, suppress_diagonal = TRUE, preserve_inactive_attention = FALSE) {
    ## calculate input matrix
    input_matrix <- matrix(1, nrow = nrow(weights), ncol = ncol(weights))
    if (suppress_diagonal) {
        diag(input_matrix) <- 0 # Create a diagonal matrix with the input vector
    }
    input_matrix <- sweep(input_matrix, 2, t(as.matrix(input)), "*") # Scale the input matrix by the input vector

    ## declare intermediate variables
    attention_prime <- if (preserve_inactive_attention) attention else attention * input

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
            stop(paste(
                "Attention gain is zero at iteration", i,
                "with attention_prime:", paste(attention_prime, collapse = ", ")
            ))
        }
        normalised <- (attention_prime * input) / attention_gain
        normalised_matrix <- sweep(input_matrix, 2, t(as.matrix(normalised)), "*") # Scale the input matrix by the normalised attention
        ## calculate model predictions based on current weights, attention and input
        predictions <- rowSums(weights * normalised_matrix * input_matrix) # Simulate predictions as the sum of weights scaled by attention and input
        if (reduced) {
            predictions[1:3] <- 0 # Only consider predictions for active input nodes
        }
        error <- teacher - predictions

        attention_shift <- AttentionShift(weights, predictions, input, error, attention_prime, attention_gain, P, rho, suppress_diagonal)

        attention_prime <- attention_prime + attention_shift * input
        ## clamp attention to be between an arbitrarily small value close to 0 and Infinity
        attention_prime <- pmax(attention_prime, 0.0001)

        attention_output <- rbind(attention_output, attention_prime)
        predictions_output <- rbind(predictions_output, t(data.matrix(predictions)))
        error_output <- rbind(error_output, t(data.matrix(error)))
        attention_delta_output <- rbind(attention_delta_output, data.matrix(attention_shift))
    }

    list(
        attention = attention_output,
        predictions = predictions_output,
        error = error_output,
        attention_delta = attention_delta_output
    )
}

#' @title attention_shift_loop_matrix
#' @description Row-wise variant of attention_shift_loop() where every output node keeps its own attention vector (a full attention matrix rather than a single shared vector), with the attention shift applied independently to each row of the weight/attention matrices.
#' @param weights A matrix of weights representing the connections between nodes.
#' @param input A vector representing the input to the model.
#' @param attention_matrix A matrix of per-row attention weights, matching the dimensions of `weights`.
#' @param P A scalar representing the power used for normalization in the attention shift.
#' @param rho A scalar representing the learning rate for the attention shift.
#' @param teacher A vector representing the target output of the model, one value per row of `weights`.
#' @param n_iterations Number of update iterations to run per row.
#' @param suppress_diagonal Passed through to AttentionShift(). Default TRUE preserves the original auto-associative behavior (self-connections suppressed). Pass FALSE for a bipartite model where stimulus features and outcome nodes are separate populations. Each row's attention is never reset by `input` mid-loop, so calling this repeatedly with `n_iterations = 1` for a rotating sequence of different inputs naturally preserves attention on nodes that are momentarily inactive.
attention_shift_loop_matrix <- function(weights, input, attention_matrix, P, rho, teacher, n_iterations = 10, suppress_diagonal = TRUE) {
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
                rho,
                suppress_diagonal
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

    list(
        attention = attention_history,
        attention_delta = attention_delta_history,
        attention_matrix_final = attention_prime,
        example_row = attention_shift_row,
        prediction = prediction_history,
        error = error_history
    )
}
