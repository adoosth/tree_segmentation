library(lidR)
library(glue)

threshold = 0.5

# Initialize vectors to store true positives, false positives and false negatives for each class
tp_50 <- numeric()
fp_50 <- numeric()
fn_50 <- numeric()
tp_25 <- numeric()
fp_25 <- numeric()
fn_25 <- numeric()
all_iou <- c()

# input las file path from command line
las_file_path <- commandArgs(trailingOnly = TRUE)[1]
focus_ratio <- as.numeric(commandArgs(trailingOnly = TRUE)[2])

# out is same as input minus extension with .txt extension
out_file_path <- glue(substr(las_file_path, 1, nchar(las_file_path) - 4), ".txt")

# Read in the las file
las <- readLAS(las_file_path)

# Center the las file in the x, y plane
las$X <- las$X - min(las$X) - (max(las$X) - min(las$X))/2
las$Y <- las$Y - min(las$Y) - (max(las$Y) - min(las$Y))/2
# print extent
print(extent(las))
# focus_side is a ratio of the side length of the focus region to the side length of the las file
focus_side <- focus_ratio * max(max(las$X) - min(las$X), max(las$Y) - min(las$Y))
# Filter points by focus region (-side/2, -side/2, side/2, side/2)
las <- filter_poi(las, X >= -focus_side/2 & X <= focus_side/2 & Y >= -focus_side/2 & Y <= focus_side/2)
print(extent(las))

thresholds <- c(0.25, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
# get arrays for each threshold
tps <- list()
fps <- list()
fns <- list()

for (threshold in thresholds) {
  tps[[as.character(threshold)]] <- numeric()
  fps[[as.character(threshold)]] <- numeric()
  fns[[as.character(threshold)]] <- numeric()
}

# Loop over each ground truth instance for small objects
for (j in unique(las$hitObjectId)) {
  # Get the bounding box of the ground truth instance
  gt <- las[las$hitObjectId == j]
  gt_bb <- bbox(gt)
  # Get the predicted instances that intersect with the ground truth instance
  pred_intersects <- filter_poi(las, X >= gt_bb[1], X <= gt_bb[3], Y >= gt_bb[2], Y <= gt_bb[4])

  # If there are no predicted instances that intersect, increment false negative by one\
  if (nrow(pred_intersects) == 0) {
    for (threshold in thresholds) {
      fns[[as.character(threshold)]] <- c(fns[[as.character(threshold)]], 1)
    }
  } else {
    # Compute the IoU for each predicted instance that intersects
    iou <- sapply(unique(pred_intersects$treeID), function(k) {
      intsect <- sum(gt$treeID == k, na.rm=T)
      onion <- nrow(gt) + sum(las$treeID == k, na.rm=T) - intsect
      intsect/onion
    })
    # Find the best matching predicted instance based on IoU
    best_match <- which.max(iou)
    all_iou <- append(all_iou, iou[best_match])

    for (threshold in thresholds) {
      if (iou[best_match] >= threshold) {
        tps[[as.character(threshold)]] <- c(tps[[as.character(threshold)]], 1)
      } else {
        fns[[as.character(threshold)]] <- c(fns[[as.character(threshold)]], 1)
      }
    }
  }
}

APs <- list()

for (threshold in thresholds) {
  
  TPs <- length(tps[[as.character(threshold)]])
  FNs <- length(fns[[as.character(threshold)]])
  FPs <- length(unique(las$treeID)) - TPs
  if (threshold == 0.25) {
    print(glue("TP@25: {TPs}"))
    print(glue("FP@25: {FPs}"))
    print(glue("FN@25: {FNs}"))
    print(glue("mIoU: {mean(all_iou) * 100}"))
    # print ious
    print(glue("IoUs: {all_iou}"))
  }
  val <- 100 * TPs / (length(unique(las$treeID)))
  # round to 2 decimal places
  APs[[as.character(threshold)]] <- round(val, 2)
}

# TP_50 = length(tps[["0.5"]])
# FN_50 = length(fns[["0.5"]])
# FP_50 = length(unique(las$hitObjectId)) - TP_50

# TP_25 = length(tps[["0.25"]])
# FN_25 = length(fns[["0.25"]])
# FP_25 = length(unique(las$hitObjectId)) - TP_25

miou = mean(all_iou) * 100
miou = round(miou, 2)

# multiply APs by 100
#APs <- lapply(APs, function(x) {x * 100})

# AP is mean average AP from 0.5 to 0.95
AP = mean(c(APs[["0.5"]], APs[["0.55"]], APs[["0.6"]], APs[["0.65"]], APs[["0.7"]], APs[["0.75"]], APs[["0.8"]], APs[["0.85"]], APs[["0.9"]], APs[["0.95"]]))
AP = round(AP, 2)
# print(glue("TP@50: ", TP_50))
# print(glue("FP@50: ", FP_50))
# print(glue("FN@50: ", FN_50))

# print(glue("TP@25: ", TP_25))
# print(glue("FP@25: ", FP_25))
# print(glue("FN@25: ", FN_25))

# print(glue("mIoU: ", 100* miou))
# print(glue("AP:   ", 100*AP))
# print(glue("AP@50: ", 100*APs[["0.5"]]))
# print(glue("AP@25: ", 100*APs[["0.25"]]))

# print as table
# print(glue("{miou},{AP},{APs[["0.5"]]},{APs[["0.25"]]}"))
print(glue("{miou}\t{AP}\t{APs[[\"0.5\"]]}\t{APs[[\"0.25\"]]}"))

# write results to file
# create file if it doesn't exist
if (!file.exists(out_file_path)) {
  file.create(out_file_path)
}
# write results to csv file

# write(glue("mIoU: ", 100* miou), out_file_path, append = TRUE)
# write(glue("AP:   ", 100*AP), out_file_path, append = TRUE)
# write(glue("AP@50: ", 100*APs[["0.5"]]), out_file_path, append = TRUE)
# write(glue("AP@25: ", 100*APs[["0.25"]]), out_file_path, append = TRUE)

# write(glue("TP@50: ", TP_50), out_file_path, append = FALSE)
# write(glue("FP@50: ", FP_50), out_file_path, append = TRUE)
# write(glue("FN@50: ", FN_50), out_file_path, append = TRUE)
# write(glue("TP@25: ", TP_25), out_file_path, append = TRUE)
# write(glue("FP@25: ", FP_25), out_file_path, append = TRUE)
# write(glue("FN@25: ", FN_25), out_file_path, append = TRUE)

csv_file_path <- glue(substr(out_file_path, 1, nchar(out_file_path) - 4), ".csv")

if (!file.exists(csv_file_path)) {
  file.create(csv_file_path)
}

write(glue("{miou},{AP},{APs[[\"0.5\"]]},{APs[[\"0.25\"]]}"), csv_file_path, append = TRUE)

