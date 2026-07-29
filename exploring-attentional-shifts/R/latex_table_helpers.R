## Helpers for exporting small, publication-ready LaTeX (booktabs) tables and
## compiling them to PDF/PNG, used to export stimulus/exemplar tables from the
## analyse-multi-outcome-*.R scripts in a style similar to how categorization
## papers present their stimulus sets (compact dimension/feedback patterns,
## booktabs rules, no vertical lines).

#' @title compile_latex_to_png
#' @description Compile a standalone LaTeX document to PDF via tinytex, then rasterize the first page to PNG via pdftoppm.
#' @param tex_content Character string of the full .tex document source.
#' @param out_path_noext Output path without extension; produces `<out_path_noext>.tex`, `.pdf` and `.png`.
#' @param dpi Resolution for the PNG rasterization.
compile_latex_to_png <- function(tex_content, out_path_noext, dpi = 400) {
    tex_file <- paste0(out_path_noext, ".tex")
    writeLines(tex_content, tex_file)

    pdf_file <- tinytex::pdflatex(tex_file, clean = TRUE)
    expected_pdf <- paste0(out_path_noext, ".pdf")
    if (!identical(normalizePath(pdf_file), normalizePath(expected_pdf, mustWork = FALSE))) {
        file.copy(pdf_file, expected_pdf, overwrite = TRUE)
    }

    png_file <- paste0(out_path_noext, ".png")
    system2("pdftoppm", c("-r", dpi, "-png", "-singlefile", expected_pdf, out_path_noext))

    invisible(list(tex = tex_file, pdf = expected_pdf, png = png_file))
}

#' @title bit_string
#' @description Format a 0/1 numeric vector as a compact digit string (e.g. c(1,0,1,0) -> "1010"), the compact pattern notation used in categorization exemplar tables.
bit_string <- function(x) {
    paste(as.integer(x), collapse = "")
}

#' @title write_exemplar_table_tex
#' @description Build a small standalone booktabs LaTeX table (Stimulus, Input Pattern, Feedback Vector) for one condition's stimulus set.
#' @param stimuli A list of stimulus/outcome pairs, each a list with elements `name`, `input`, `teacher`.
write_exemplar_table_tex <- function(stimuli) {
    rows <- vapply(stimuli, function(s) {
        sprintf("%s & %s & %s \\\\", s$name, bit_string(s$input), bit_string(s$teacher))
    }, character(1))

    paste(
        "\\documentclass[preview,border=4pt]{standalone}",
        "\\usepackage{booktabs}",
        "\\begin{document}",
        "\\begin{tabular}{ccc}",
        "\\toprule",
        "Stimulus & Input Pattern & Feedback Vector \\\\",
        "\\midrule",
        paste(rows, collapse = "\n"),
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{document}",
        sep = "\n"
    )
}
