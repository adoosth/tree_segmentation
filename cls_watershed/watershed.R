library(lidR)
library(glue)
# library(future)
# plan(multisession, workers = 4)

glue <- glue::glue

#data_dir <- "/home/adoos/Downloads/lazes/classic_exp1/"
#las_file <- "20230725-164412-synforest-example-ULS.laz"
#las_file <- "synforest-sparse-multi-20230705-094907.laz"
#las_file <- "20230830-235058-syn-discrete-3-ULS.laz"
#las_file <- "20230725-152622-synforest-example-ULS-no-leaves.laz"
#las_file <- "2022-05-17___F103_RE_Data_1_new_points_AllBands_SOR_0.01.laz"

# input las file path from command line
# las_file_path <- "/data/full-syn-multi-thin/train/syn-multi-thin-1-ULS.laz"
data_dir <- commandArgs(trailingOnly = TRUE)[1]
dataset <- commandArgs(trailingOnly = TRUE)[2]
batch_id <- as.numeric(commandArgs(trailingOnly = TRUE)[3])

path_to_csv <- glue("/data/{data_dir}/{dataset}.csv")
# read csv file of boxes (forest_id, x, y, width, height)
df <- read.csv(path_to_csv, header = TRUE, sep = ",")

# get all forest_ids
forest_ids <- unique(df$forest_id)

# get batch_id th row of df
box <- df[batch_id,]
laz_dir <- glue("/data/{data_dir}/{box$forest_id}/")

print("Checking tiles...") # end with .laz or .las
tiles <- list.files(laz_dir, pattern = ".laz", full.names = TRUE) # TODO
# sort tiles by name
tiles <- sort(tiles)

# get extants of all tiles as data frame
extents <- lapply(tiles, function(x) {
  las <- readLAScatalog(x)
    # las$Min.X and las$Max.X and las$Min.Y and las$Max.Y
    return (list(x, las$Min.X, las$Max.X, las$Min.Y, las$Max.Y))
    #return (list(las$extent@xmin, las$extent@xmax, las$extent@ymin, las$extent@ymax))
})
extents <- do.call(rbind, extents)
extents <- as.data.frame(extents)
colnames(extents) <- c("path", "xmin", "xmax", "ymin", "ymax")

# only load tiles that intersect with box
extents <- extents[extents$xmin < box$x + box$width & extents$xmax > box$x & extents$ymin < box$y + box$height & extents$ymax > box$y,]

# get all tiles that intersect with box
tiles <- extents$path
print(tiles)
# read tiles in a loop
print("Load tiles from list")
raw_tiles <- lapply(tiles, function(x) {
  las <- readLAScatalog(x)
  return (las)
})

raw_las <- do.call(rbind, raw_tiles)
# load raw_las
raw_las <- readLAS(raw_las)

# filter points by box with filterpoi with conditions (raw_las, X >= box$x & X <= box$x + box$width & Y >= box$y & Y <= box$y + box$height)
las <- filter_poi(raw_las, X >= box$x & X <= box$x + box$width & Y >= box$y & Y <= box$y + box$height)

print("Reprojecting las file")
las <- lidR::las_reoffset(las, xoffset = 0)
lidR::projection(las) <- sp::CRS("+init=epsg:32633")

print("Computing CHM")
# 0.5 m resolution
chm_p2r_05 <- rasterize_canopy(las, 0.5,, p2r(subcircle = 0.2), pkg = "terra")
# create output file path
save_path <- glue("results/{data_dir}/{dataset}-{batch_id}-chm_p2r_05.png")
# create folder if it doesn't exist
dir.create(glue("results/{data_dir}"), showWarnings = FALSE, recursive = TRUE)
# write chm to file
png(filename=save_path, width = 1000, height = 1000, res = 200)
plot(chm_p2r_05)
dev.off()

# Post-processing median filter
kernel <- matrix(1,3,3)
chm_p2r_05_smoothed <- terra::focal(chm_p2r_05, w = kernel, fun = median, na.rm = TRUE)
save_path <- glue("results/{data_dir}/{dataset}-{batch_id}-chm_p2r_05_smoothed.png")
png(filename=save_path, width = 1000, height = 1000, res = 200)
plot(chm_p2r_05_smoothed)
dev.off()
# chm_p2r_1_smoothed <- terra::focal(chm_p2r_1, w = kernel, fun = median, na.rm = TRUE)

# plot(chm_p2r_05_smoothed)

crowns <- watershed(chm_p2r_05_smoothed)()

algo <-watershed(chm_p2r_05_smoothed)

# add tree ID column to las same as hitObjectId with add_attribute
# las$treeID <- las$hitObjectId doesn't work because adding a new object is forbidden
# las <- add_attribute(las, "treeID", las$hitObjectId) # las$hitObjectId is not a string
print(names(las))
if (!("hitObjectId" %in% names(las))) {
  print("Adding hitObjectId attribute")
  las <- add_lasattribute(las, las$PointSourceID, "hitObjectId", "Ground truth") # What does this do?
}
las <- add_lasattribute(las, las$hitObjectId, "treeID", "Watershed")

segmented <- segment_trees(las, algo)

# reindex segmented tree IDs to start at 1 and be sequential and make NA values 0
segmented$treeID <- as.numeric(factor(segmented$treeID))
segmented$treeID[is.na(segmented$treeID)] <- 0
segmented$treeID <- as.integer(segmented$treeID)
las$treeID <- segmented$treeID

out_file_path <- glue("results/{data_dir}/{dataset}-{batch_id}-segmented.laz")
writeLAS(las, out_file_path)

#print("Plotting to file")
#png(filename="./results/trees.png", width = 1000, height = 1000, res = 300)
#plot(segmented, bg = "white", size = 4, color = "treeID") # visualize trees
#dev.off()
#print("Done")

# typeof(las) is S4
# typeof(segmented) is S4


