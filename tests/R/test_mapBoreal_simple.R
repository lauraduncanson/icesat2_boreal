library(testthat)

# TODO: update the local path once CI/CD is up and running.

source("~/src/icesat2_boreal/lib/mapBoreal_simple.R")

# Sample tile_data for testing
tile_data <- data.frame(
  doy = c(50, 100, 150, 200, 250, 300, 365),
  solar_elevation = c(0, 0, 1, 2, 3, 4, 5)
)

test_that("DOY_and_solar_filter works correctly", {
  result <- DOY_and_solar_filter(tile_data, start_DOY = 100, end_DOY = 250, solar_elevation = 5)
  print(result)
  expect_equal(result, c(2, 3, 4, 5))
})

test_that("late_season_filter returns correct filter when enough samples found early", {
  result <- late_season_filter(tile_data, minDOY = 50, maxDOY = 150, min_icesat2_samples = 3, max_sol_el = 5)
  
  expect_equal(result$filter, c(1, 2, 3))
  expect_equal(result$late_season_DOY, 150)
})

test_that("late_season_filter returns correct filter when extending DOY", {
  result <- late_season_filter(tile_data, minDOY = 50, maxDOY = 150, min_icesat2_samples = 5, max_sol_el = 5)

  expect_equal(result$filter, c(1, 2, 3, 4))
  expect_equal(result$late_season_DOY, 150 + 3 * 30)  # maxDOY + 3 months
})

test_that("early_and_late_season_filter returns correct filter when enough samples found early", {
  result <- early_and_late_season_filter(tile_data, minDOY = 100, late_season_DOY = 200, min_icesat2_samples = 3, max_sol_el = 5)
  
  expect_equal(result$filter, c(2, 3, 4))
  expect_equal(result$early_season_DOY, 100)  # maxDOY + 3 months
})


test_that("early_and_late_season_filter returns correct filter when extending DOY", {
  result <- early_and_late_season_filter(tile_data, minDOY = 200, late_season_DOY = 250, min_icesat2_samples = 5, max_sol_el = 5)
  
  expect_equal(result$filter, c(3, 4, 5))
  expect_equal(result$early_season_DOY, 110)  # minDOY - 3 months
})

test_that("expand_training_around_growing_season returns the correct data when enough samples found early with 0 solar elevation", {
  result <- expand_training_around_growing_season(tile_data, minDOY = 50, maxDOY = 150, max_sol_el = 5, min_icesat2_samples = 2)
  
  expect_equal(result, tile_data[c(1, 2), ])
})

test_that("expand_training_around_growing_season returns the correct data when enough samples found early with max solar elevation", {
  result <- expand_training_around_growing_season(tile_data, minDOY = 50, maxDOY = 150, max_sol_el = 5, min_icesat2_samples = 3)
  
  expect_equal(result, tile_data[c(1, 2, 3), ])
})

test_that("expand_training_around_growing_season returns the correct data when extending DOY", {
  result <- expand_training_around_growing_season(tile_data, minDOY = 50, maxDOY = 150, max_sol_el = 5, min_icesat2_samples = 4)
  
  expect_equal(result, tile_data[c(1, 2, 3, 4), ])
})

test_that("expand_training_around_growing_season returns the correct data when extending both minDOY and maxDOY", {
  result <- expand_training_around_growing_season(tile_data, minDOY = 100, maxDOY = 150, max_sol_el = 5, min_icesat2_samples = 4)
  expect_equal(result, tile_data[c(1, 2, 3, 4), ])
})

test_that("expand_training_around_growing_season returns the correct data when extending doesn't give enough samples", {
  result <- expand_training_around_growing_season(tile_data, minDOY = 100, maxDOY = 150, max_sol_el = 5, min_icesat2_samples = 5)

  # Note that we can't find 5 samples in this case, but this is the best we can do
  expect_equal(result, tile_data[c(1, 2, 3, 4), ])
})
