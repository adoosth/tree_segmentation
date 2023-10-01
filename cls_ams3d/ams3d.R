library(glue)
#library(tidyverse)
#library(raster)
library(lidR)
library(crownsegmentr)
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

# create folder if it doesn't exist
dir.create(glue("results/{data_dir}"), showWarnings = FALSE, recursive = TRUE)

# plot(chm_p2r_05_smoothed)

#crowns <- watershed(chm_p2r_05_smoothed)()

#algo <-watershed(chm_p2r_05_smoothed)

# add tree ID column to las same as hitObjectId with add_attribute
# las$treeID <- las$hitObjectId doesn't work because adding a new object is forbidden
# las <- add_attribute(las, "treeID", las$hitObjectId) # las$hitObjectId is not a string
print(names(las))
if (!("hitObjectId" %in% names(las))) {
  print("Adding hitObjectId attribute")
  las <- add_lasattribute(las, las$PointSourceID, "hitObjectId", "Ground truth") # What does this do?
}
las <- add_lasattribute(las, las$hitObjectId, "treeID", "AMS3D")

homogenized_points <- lidR::decimate_points(las, lidR::homogenize(30, res = 0.2)) # Adjust resolution

start_time <- Sys.time()
segmented_points <- crownsegmentr::segment_tree_crowns(
  point_cloud = homogenized_points,
  crown_diameter_2_tree_height = 0.40,
  crown_height_2_tree_height = 0.40,
)
Sys.time() - start_time

# segmented_points$hitObjectId <- segmented_points$crown_id
las <- add_attribute(homogenized_points, segmented_points$crown_id, "treeID")

print(glue("Found labels ", data.table::uniqueN(segmented_points@data$crown_id)))
print(glue("Original trees ", data.table::uniqueN(las@data$hitObjectId)))

# reindex segmented tree IDs to start at 1 and be sequential and make NA values 0
segmented_points$crown_id <- as.numeric(factor(segmented_points$crown_id))
segmented_points$crown_id[is.na(segmented_points$crown_id)] <- 0
las$treeID <- as.integer(segmented_points$crown_id)
#print(las$treeID)

#setClass(
#  "LAS_bbox",
#  contains = "LAS",
#  slots = c(bbox = "matrix")
#)

# raw_raw_las <- lidR::readLAS(las_file_path)
# raw_las <- new("LAS_bbox")
# slot(raw_las, "header") <- slot(raw_raw_las, "header")
# slot(raw_las, "data") <- slot(raw_raw_las, "data")

# header <- as.list(raw_las@header)
# new_header <- rlas::header_update(header, raw_las@data)
# new_header <- lidR::LASheader(new_header)
# raw_las@header <- new_header
# las_extent <- extent(raw_las)
# bbox <- matrix(
#   c(las_extent@xmin, las_extent@ymin, las_extent@xmax, las_extent@ymax),
#   nrow = 2,
#   byrow = TRUE,
#   dimnames = list(c("x", "y"), c("min", "max"))
# )
# slot(raw_las, "bbox") <- bbox
# raw_las@bbox[1,1] <- new_header@PHB[["Min X"]]
# raw_las@bbox[1,2] <- new_header@PHB[["Max X"]]
# raw_las@bbox[2,1] <- new_header@PHB[["Min Y"]]
# raw_las@bbox[2,2] <- new_header@PHB[["Max Y"]]

# raw_las <- lidR::las_reoffset(raw_las, xoffset = 0)

# lidR::projection(raw_las) <- sp::CRS("+init=epsg:32633")

# x_min <- floor(raw_las@bbox["x", "min"])
# y_min <- floor(raw_las@bbox["y", "min"])

# homogenized_points <- raw_las %>%
#   lidR::decimate_points(., lidR::homogenize(2, res = 1)) # Adjust resolution

# homogenized_points


out_file_path <- glue("results/{data_dir}/{dataset}-{batch_id}-ams3d.laz")
writeLAS(las, out_file_path)