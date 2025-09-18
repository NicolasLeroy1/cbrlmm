#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(parallel)
})

script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- args[grep("^--file=", args)]
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/", mustWork = TRUE))
  }
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(normalizePath(sys.frames()[[1]]$ofile, winslash = "/", mustWork = TRUE))
  }
  stop("Unable to determine script path")
}

script_file <- script_path()
script_dir <- dirname(script_file)
source(file.path(script_dir, "simulation.R"))

compare_opt_name <- "brlmm.run_compare_on_source"
compare_opt_old <- getOption(compare_opt_name)
options(brlmm.run_compare_on_source = FALSE)
on.exit(options(brlmm.run_compare_on_source = compare_opt_old), add = TRUE)
source(file.path(script_dir, "..", "tests", "test_compare.R"))

run_fold <- function(sim_data, fold_indices, config) {
  train_idx <- fold_indices$train
  test_idx <- fold_indices$test

  fold_problem <- list(
    y = sim_data$y[train_idx],
    X_list = NULL,
    K_list = NULL
  )

  if (!is.null(sim_data$X_list)) {
    fold_problem$X_list <- lapply(sim_data$X_list, function(X) X[train_idx, , drop = FALSE])
  }
  if (!is.null(sim_data$K_list)) {
    fold_problem$K_list <- lapply(sim_data$K_list, function(K) K[train_idx, train_idx, drop = FALSE])
  }

  build_native()
  set.seed(config$seed)
  time_r <- system.time({
    fit_r <- run_brlmm_r(fold_problem$y, fold_problem$X_list, fold_problem$K_list,
                         n_iter = config$n_iter, burnin = config$burnin,
                         thinning = config$thinning,
                         inertia_threshold = config$inertia_threshold,
                         scale_X = config$scale_X)
  })
  new_X_list <- if (!is.null(sim_data$X_list)) {
    lapply(sim_data$X_list, function(X) X[test_idx, , drop = FALSE])
  } else NULL
  new_K_list <- if (!is.null(sim_data$K_list)) {
    lapply(sim_data$K_list, function(K) K[test_idx, train_idx, drop = FALSE])
  } else NULL

  prediction_r <- chain_predict(fit_r,
                                new_X_list = new_X_list,
                                new_K_list = new_K_list)
  mean_pred_r <- rowMeans(prediction_r)

  set.seed(config$seed)
  time_c <- system.time({
    fit_c <- run_brlmm_c(fold_problem$y, fold_problem$X_list, fold_problem$K_list,
                         n_iter = config$n_iter, burnin = config$burnin,
                         thinning = config$thinning,
                         inertia_threshold = config$inertia_threshold,
                         scale_X = config$scale_X)
  })
  if (!is.null(fit_c$metadata)) {
    fit_c$LX <- fit_c$metadata$LX
    fit_c$LK <- fit_c$metadata$LK
    fit_c$n <- fit_c$metadata$n
  }
  if (!is.null(fit_c$latent_list)) {
    fit_c$latent_list <- lapply(fit_c$latent_list, function(latent) {
      if (is.null(latent$sd) && !is.null(latent$scale)) {
        latent$sd <- latent$scale
      }
      latent
    })
  }
  prediction_c <- chain_predict(fit_c,
                                new_X_list = new_X_list,
                                new_K_list = new_K_list)
  mean_pred_c <- rowMeans(prediction_c)

  y_true <- sim_data$y[test_idx]
  residual_r <- y_true - mean_pred_r
  residual_c <- y_true - mean_pred_c

  cor_r <- if (sd(mean_pred_r) > 0) cor(mean_pred_r, y_true)^2 else NA_real_
  cor_c <- if (sd(mean_pred_c) > 0) cor(mean_pred_c, y_true)^2 else NA_real_
  r2_r <- 1 - sum(residual_r^2) / sum((y_true - mean(y_true))^2)
  r2_c <- 1 - sum(residual_c^2) / sum((y_true - mean(y_true))^2)

  list(
    cor_r = cor_r,
    cor_c = cor_c,
    r2_r = r2_r,
    r2_c = r2_c,
    time_r = time_r["elapsed"],
    time_c = time_c["elapsed"]
  )
}

create_folds <- function(n, k) {
  fold_ids <- sample(rep(1:k, length.out = n))
  lapply(1:k, function(f) {
    list(
      train = which(fold_ids != f),
      test = which(fold_ids == f)
    )
  })
}

benchmark_once <- function(config) {
  sim_data <- simulation_brlmm(LX = config$LX, LK = config$LK,
                               n_train = config$n_train, n_test = config$n_test,
                               k = config$k, r2 = config$r2, p = config$p)

  folds <- create_folds(length(sim_data$y), config$folds)

  fold_results <- lapply(folds, run_fold, sim_data = sim_data, config = config)

  cor_r <- mean(sapply(fold_results, "[[", "cor_r"), na.rm = TRUE)
  cor_c <- mean(sapply(fold_results, "[[", "cor_c"), na.rm = TRUE)
  r2_r <- mean(sapply(fold_results, "[[", "r2_r"), na.rm = TRUE)
  r2_c <- mean(sapply(fold_results, "[[", "r2_c"), na.rm = TRUE)
  time_r <- mean(sapply(fold_results, "[[", "time_r"))
  time_c <- mean(sapply(fold_results, "[[", "time_c"))

  data.frame(
    LX = config$LX,
    LK = config$LK,
    n_train = config$n_train,
    n_test = config$n_test,
    k = config$k,
    r2 = config$r2,
    p = config$p,
    cor_r = cor_r,
    cor_c = cor_c,
    r2_r = r2_r,
    r2_c = r2_c,
    time_r = time_r,
    time_c = time_c
  )
}

random_config <- function(seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  list(
    LX = sample(1:500, 1),
    LK = 0L,
    n_train = sample(50:2000, 1),
    n_test = sample(40:100, 1),
    k = sample(3:8, 1),
    r2 = runif(1, 0.5, 0.9),
    p = sample(10:40, 1),
    folds = 5,
    n_iter = 2000L,
    burnin = 500L,
    thinning = 5L,
    inertia_threshold = 0.95,
    scale_X = TRUE,
    seed = sample.int(.Machine$integer.max, 1)
  )
}

main <- function(iterations = 10) {
  configs <- replicate(iterations, random_config(), simplify = FALSE)
  results <- do.call(rbind, lapply(configs, benchmark_once))
  print(results)
}

if (identical(environment(), globalenv())) {
  main()
}
