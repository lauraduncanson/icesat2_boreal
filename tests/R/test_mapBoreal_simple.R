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

test_that("reduce_sample_size returns correct sample size", {

  tile_data <- data.frame(A = 1:10, B = 11:20)
  sample_size <- 5
  result <- reduce_sample_size(tile_data, sample_size)

  expect_equal(nrow(result), sample_size)
})

test_that("reduce_sample_size samples valid rows", {

  df <- data.frame(A = 1:10, B = 11:20)
  sample_size <- 5
  result <- reduce_sample_size(df, sample_size)

  expect_true(all(result$A %in% df$A))
})

test_that("reduce_sample_size handles edge cases", {

  df <- data.frame(A = 1:10, B = 11:20)

  # Test when sample size is the total number of rows
  result <- reduce_sample_size(df, 10)
  expect_equal(nrow(result), 10)

  # Test when sample size is larger than available rows
  expect_error(reduce_sample_size(df, 15),
               "cannot take a sample larger than the population when 'replace = FALSE'")
})

test_that("remove_stale_columns works as expected", {

  df <- data.frame(a = 1:3, b = 4:6, c = 7:9)
  result <- remove_stale_columns(df, c("a"))

  expected <- data.frame(b = 4:6, c = 7:9)
  expect_true(identical(result, expected))
})

test_that("remove_stale_columns works as expected when removing multiple columns that exist", {

  df <- data.frame(a = 1:3, b = 4:6, c = 7:9)
  result <- remove_stale_columns(df, c("a", "c"))

  expected <- data.frame(b = 4:6)
  expect_true(identical(result, expected))
})

test_that("remove_stale_columns works as expected when removing a column that does not exist", {
  df <- data.frame(a = 1:3, b = 4:6, c = 7:9)
  result <- remove_stale_columns(df, c("d"))

  expect_equal(result, df)  # No change expected
})

test_that("remove_stale_columns works as expected when removing both existing and non-existing columns", {

  df <- data.frame(a = 1:3, b = 4:6, c = 7:9)
  result <- remove_stale_columns(df, c("a", "d"))

  expected <- data.frame(b = 4:6, c = 7:9)
  expect_true(identical(result, expected))
})

test_that("remove_stale_columns works as expected when no columns to remove (empty column_names)", {

  df <- data.frame(a = 1:3, b = 4:6, c = 7:9)
  result <- remove_stale_columns(df, character(0))

  expect_equal(result, df)  # No change expected
})

test_that("remove_stale_columns works as expected when  empty data frame", {

  empty_df <- data.frame()
  result <- remove_stale_columns(empty_df, c("a"))
  expect_equal(result, empty_df)  # No change expected for empty data frame
})

test_that("remove_stale_columns works as expected when Data frame with only one column, removing that column", {

  single_col_df <- data.frame(a = 1:3)
  # dropping column a doesn't remove the row names, hence check.attributes=False below
  result <- remove_stale_columns(single_col_df, c("a"))

  expected <- data.frame()  # Expect an empty data frame
  expect_true(isTRUE(all.equal(result, expected, check.attributes = FALSE)))
})

test_that("remove_stale_columns works as expected when Non-data frame input (testing if error is handled)", {

  expect_error(remove_stale_columns(list(a = 1:3), c("a")), "incorrect number of dimensions")
})
