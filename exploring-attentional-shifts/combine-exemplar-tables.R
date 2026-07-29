## Combine the per-condition exemplar tables exported by the three
## analyse-multi-outcome-*.R scripts (non-overlapping, overlapping-all,
## overlapping-features) into a single publication-ready LaTeX table.
##
## Reads the raw stimuli lists each script saved as tables/stimuli_<condition>.rds
## (rather than re-running the simulations) and lays them out the way
## categorization papers present their stimulus sets: one row per stimulus,
## its input (dimension) pattern, and its feedback vector under each overlap
## condition, grouped under a shared "Feedback Vector" header.
##
## Run from the repository root, after the three analyse-multi-outcome-*.R
## scripts have each been run at least once. Produces
## tables/exemplar_table_combined.tex/.pdf/.png.

project_dir <- "exploring-attentional-shifts"
tables_dir <- file.path(project_dir, "tables")
source(file.path(project_dir, "R", "latex_table_helpers.R"))

conditions <- c(
    "Non-overlapping" = "non_overlapping",
    "Overlapping (All)" = "overlapping_all",
    "Overlapping (Features)" = "overlapping_features"
)

stimuli_by_condition <- lapply(conditions, function(condition_name) {
    rds_path <- file.path(tables_dir, paste0("stimuli_", condition_name, ".rds"))
    if (!file.exists(rds_path)) {
        stop("Missing ", rds_path, " -- run analyse-multi-outcome-", gsub("_", "-", condition_name), ".R first")
    }
    readRDS(rds_path)
})

## the three conditions are expected to share the same stimulus names and
## input patterns (only the feedback vectors differ) -- verify that rather
## than silently combining mismatched rows
reference <- stimuli_by_condition[[1]]
stimulus_names <- vapply(reference, `[[`, character(1), "name")
reference_inputs <- lapply(reference, `[[`, "input")

for (condition_label in names(stimuli_by_condition)[-1]) {
    this_stimuli <- stimuli_by_condition[[condition_label]]
    this_names <- vapply(this_stimuli, `[[`, character(1), "name")
    if (!identical(this_names, stimulus_names)) {
        stop("Stimulus names differ for condition '", condition_label, "': expected ", paste(stimulus_names, collapse = ","), " got ", paste(this_names, collapse = ","))
    }
    this_inputs <- lapply(this_stimuli, `[[`, "input")
    if (!identical(this_inputs, reference_inputs)) {
        stop("Input patterns differ for condition '", condition_label, "' -- the combined table assumes identical inputs across conditions")
    }
}

## ---- Build the combined table body -------------------------------------------
n_conditions <- length(conditions)
body_rows <- vapply(seq_along(stimulus_names), function(i) {
    input_str <- bit_string(reference[[i]]$input)
    feedback_strs <- vapply(stimuli_by_condition, function(s) bit_string(s[[i]]$teacher), character(1))
    paste(c(stimulus_names[i], input_str, feedback_strs), collapse = " & ")
}, character(1))

header_labels <- paste(names(conditions), collapse = " & ")
col_spec <- paste0("cc", strrep("c", n_conditions))

tex_content <- paste(
    "\\documentclass[preview,border=8pt]{standalone}",
    "\\usepackage{booktabs}",
    "\\begin{document}",
    "{\\bfseries Table 1}\\\\",
    "{\\itshape Stimulus Input Patterns and Feedback Vectors Across Overlap Conditions}\\\\[6pt]",
    sprintf("\\begin{tabular}{%s}", col_spec),
    "\\toprule",
    sprintf(" & & \\multicolumn{%d}{c}{Feedback Vector} \\\\", n_conditions),
    sprintf("\\cmidrule(lr){3-%d}", 2 + n_conditions),
    sprintf("Stimulus & Input Pattern & %s \\\\", header_labels),
    "\\midrule",
    paste(body_rows, collapse = " \\\\\n"),
    " \\\\",
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{document}",
    sep = "\n"
)

result <- compile_latex_to_png(tex_content, file.path(tables_dir, "exemplar_table_combined"))
print(result)
