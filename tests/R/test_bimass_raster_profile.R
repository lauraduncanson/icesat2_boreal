library(terra)
library(testthat)

compare_geotiffs <- function(file1, file2) {
  raster1 <- rast(file1)
  raster2 <- rast(file2)
  
  if (!all(dim(raster1) == dim(raster2))) {
    print('dims dont match')
    return(FALSE)
  }
  
  if (!all(ext(raster1) == ext(raster2))) {
    print('extents dont match')
    return(FALSE)
  }
  
  if (!all(crs(raster1) == crs(raster2))) {
    print('crs dont match')
    return(FALSE)
  }
  
  vals1 <- values(raster1)
  vals2 <- values(raster2)
  
  if (!all(is.na(vals1) == is.na(vals2))) {
    print('NAs are not in the same place')
    return(FALSE)
  }
  
  if (!all(vals1[!is.na(vals1)] == vals2[!is.na(vals2)])) {
    print('not all values are equal')
    return(FALSE)
  }
  
  return(TRUE)
}

test_that("The new AGB raster matches a known good ABG raster, will update the paths once CI/CD is up...", {
  expect_true(compare_geotiffs("~/output/boreal_agb_202409171726607141_034673.tif",
                               "~/output_run3/boreal_agb_202409161726517545_034673.tif"))
  
})

