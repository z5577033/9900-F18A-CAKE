suppressMessages({
  library("limma", quiet = TRUE)
  library("arrow", quiet = TRUE)
})

runDM <- function(dataFile, designFile) {
  data <- read_feather(dataFile)
  designdf <- read_feather(designFile)

  # Coerce to data.frame
  data <- as.data.frame(data, stringsAsFactors = FALSE)
  designdf <- as.data.frame(designdf, stringsAsFactors = FALSE)

  # Expect sampleId and cancerType columns
  if (!("sampleId" %in% names(designdf))) {
    stop("designFile must contain 'sampleId'")
  }
  if (!("cancerType" %in% names(designdf))) {
    stop("designFile must contain 'cancerType'")
  }

  # Remove helper/name columns from data, keep features + sampleId
  if (!("sampleId" %in% names(data))) {
    stop("dataFile must contain 'sampleId'")
  }
  feature_cols <- setdiff(names(data), c("sampleId", "Name"))
  # data rows = samples, columns = features; index by sampleId
  rownames(data) <- as.character(data[["sampleId"]])
  data <- data[, feature_cols, drop = FALSE]

  # Coerce to numeric, robustly (limma expects numeric matrix)
  data[] <- lapply(data, function(x) suppressWarnings(as.numeric(x)))

  # Align design rows to data columns after transpose
  # After transpose, columns will be sampleIds
  # So we build a common set, then order design to match colnames(data_t)
  common_ids <- intersect(rownames(data), designdf$sampleId)
  if (length(common_ids) < 2) {
    stop("Not enough overlapping samples between data and design")
  }
  data <- data[common_ids, , drop = FALSE]
  designdf <- designdf[match(common_ids, designdf$sampleId), , drop = FALSE]

  # Transpose: rows = probes, cols = samples
  data_t <- t(as.matrix(data))

  # Group labels (length must equal ncol(data_t))
  group <- factor(designdf$cancerType)

  # Design matrix
  design <- model.matrix(~ group)

  fit <- lmFit(data_t, design)
  fit <- eBayes(fit, robust = TRUE)

  # coef 2 corresponds to the group effect in ~ group
  top <- topTable(fit, coef = 2, number = 50)

  # Return a data.frame with rownames as probe IDs (limma puts probe IDs as rownames)
  top
}
