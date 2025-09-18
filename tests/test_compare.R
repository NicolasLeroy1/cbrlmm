#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(utils)
})

# Resolve project root ---------------------------------------------------
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
project_root <- normalizePath(file.path(dirname(script_file), ".."), winslash = "/", mustWork = TRUE)
src_dir <- file.path(project_root, "src")
dll_name <- file.path(src_dir, paste0("brlmm_port", .Platform$dynlib.ext))

# Load R implementation --------------------------------------------------
source(file.path(project_root, "brlmm.R"))

build_native <- function(force = FALSE) {
  if (force || !file.exists(dll_name)) {
    message("Compiling native BRLMM port ...")
    cmd <- c("CMD", "SHLIB", "-o", dll_name,
             file.path(src_dir, "brlmm_bridge.c"),
             file.path(src_dir, "brlmm.c"),
             file.path(src_dir, "brlmm_utils.c"))
    status <- system2(file.path(R.home("bin"), "R"), cmd)
    if (!identical(status, 0L)) {
      stop("Compilation of brlmm_port shared library failed", call. = FALSE)
    }
  }
  if (!is.loaded("C_brlmm_run")) {
    dyn.load(dll_name)
  }
  invisible(dll_name)
}

run_brlmm_c <- function(y, X_list = NULL, K_list = NULL,
                        n_iter = 5000L, burnin = 1000L, thinning = 5L,
                        inertia_threshold = 0.95, scale_X = TRUE) {
  build_native()
  if (is.matrix(X_list)) {
    X_list <- list(X_list)
  }
  if (is.matrix(K_list)) {
    K_list <- list(K_list)
  }
  y <- as.numeric(y)
  if (!is.null(X_list)) {
    X_list <- lapply(X_list, function(mat) {
      storage.mode(mat) <- "double"
      mat
    })
  }
  if (!is.null(K_list)) {
    K_list <- lapply(K_list, function(mat) {
      storage.mode(mat) <- "double"
      mat
    })
  }

  res <- .Call("C_brlmm_run",
               y,
               if (is.null(X_list)) NULL else X_list,
               if (is.null(K_list)) NULL else K_list,
               as.integer(n_iter),
               as.integer(burnin),
               as.integer(thinning),
               as.numeric(inertia_threshold),
               as.logical(scale_X))

  n_sample <- length(res$mu_vec)
  res$effect_mean_list <- lapply(seq_along(res$nu_frame_list), function(l) {
    nu_mat <- res$nu_frame_list[[l]]
    if (!is.matrix(nu_mat) || nrow(nu_mat) == 0 || n_sample == 0) {
      return(matrix(0, nrow = nrow(nu_mat), ncol = 1))
    }
    lambda_vec <- res$lambda_frame[l, ]
    (nu_mat %*% matrix(lambda_vec, ncol = 1)) / n_sample
  })
  res
}

run_brlmm_r <- function(y, X_list = NULL, K_list = NULL,
                        n_iter = 5000L, burnin = 1000L, thinning = 5L,
                        inertia_threshold = 0.95, scale_X = TRUE) {
  if (is.matrix(X_list)) {
    X_list <- list(X_list)
  }
  if (is.matrix(K_list)) {
    K_list <- list(K_list)
  }
  brlm(y = y,
       X_list = X_list,
       K_list = K_list,
       niter = n_iter,
       burnin = burnin,
       thinning = thinning,
       inertia_threshold = inertia_threshold,
       method = "horseshoe",
       X_matrix_scaling = scale_X)
}

simulate_problem <- function(n = 60L,
                             X_dims = c(8L, 5L,7L,10L,20L),
                             K_dims = c(20L)) {
  stopifnot(length(X_dims) > 0L || length(K_dims) > 0L)
  X_list <- if (length(X_dims) > 0L) {
    lapply(X_dims, function(p) matrix(rnorm(n * p), nrow = n, ncol = p))
  } else NULL
  K_list <- if (length(K_dims) > 0L) {
    lapply(K_dims, function(q) {
      Z <- matrix(rnorm(n * q), nrow = n, ncol = q)
      tcrossprod(Z) / q
    })
  } else NULL
  beta_true <- rnorm(sum(X_dims), sd = 0.5)
  y <- numeric(n)
  offset <- 0L
  if (!is.null(X_list)) {
    for (idx in seq_along(X_list)) {
      p <- ncol(X_list[[idx]])
      coef_slice <- beta_true[(offset + 1L):(offset + p)]
      offset <- offset + p
      y <- y + X_list[[idx]] %*% coef_slice
    }
  }
  if (!is.null(K_list)) {
    for (K in K_list) {
      u <- rnorm(n, sd = 0.7)
      y <- y + K %*% u
    }
  }
  y <- as.numeric(y + rnorm(n, sd = 0.5))
  list(y = y, X_list = X_list, K_list = K_list)
}

chain_stats <- function(vec) {
  vec <- as.numeric(vec)
  vec <- vec[is.finite(vec)]
  if (!length(vec)) {
    return(c(mean = NA_real_, var = NA_real_))
  }
  c(mean = mean(vec), var = if (length(vec) > 1L) var(vec) else 0)
}

summarise_result <- function(res) {
  list(
    mu = chain_stats(res$mu_vec),
    sigma2 = chain_stats(res$sigma2_vec),
    lambda = chain_stats(as.numeric(res$lambda_frame)),
    nu = chain_stats(unlist(res$nu_frame_list, use.names = FALSE)),
    mse = chain_stats(res$mean_squared_error_vec),
    loss = chain_stats(res$loss_vector)
  )
}

diff_summary <- function(stats_r, stats_c) {
  keys <- names(stats_r)
  out <- lapply(keys, function(k) {
    r <- stats_r[[k]]
    c <- stats_c[[k]]
    names(r) <- names(c) <- c("mean", "var")
    c(
      R_mean = r["mean"],
      C_mean = c["mean"],
      mean_diff = r["mean"] - c["mean"],
      mean_abs_diff = abs(r["mean"] - c["mean"]),
      R_var = r["var"],
      C_var = c["var"],
      var_diff = r["var"] - c["var"],
      var_abs_diff = abs(r["var"] - c["var"])
    )
  })
  names(out) <- keys
  out
}

compute_latent_correlation <- function(latents_r, latents_c) {
  L <- min(length(latents_r), length(latents_c))
  cor_values <- vector("list", L)
  for (l in seq_len(L)) {
    U_r <- latents_r[[l]]$U
    U_c <- latents_c[[l]]$U
    if (!is.matrix(U_r) || !is.matrix(U_c)) {
      cor_values[[l]] <- NA_real_
      next
    }
    common_rows <- min(nrow(U_r), nrow(U_c))
    common_cols <- min(ncol(U_r), ncol(U_c))
    if (common_rows == 0 || common_cols == 0) {
      cor_values[[l]] <- numeric(0)
      next
    }
    U_r_sub <- U_r[seq_len(common_rows), seq_len(common_cols), drop = FALSE]
    U_c_sub <- U_c[seq_len(common_rows), seq_len(common_cols), drop = FALSE]
    cor_vec <- numeric(common_cols)
    for (j in seq_len(common_cols)) {
      cor_vec[j] <- suppressWarnings(stats::cor(U_r_sub[, j], U_c_sub[, j]))
    }
    cor_values[[l]] <- cor_vec
  }
  raw_values <- suppressWarnings(as.numeric(unlist(cor_values, use.names = FALSE)))
  values <- abs(raw_values)
  values <- values[!is.na(values)]
  summary <- if (length(values)) {
    c(min = min(values, na.rm = TRUE),
      median = stats::median(values, na.rm = TRUE),
      mean = mean(values, na.rm = TRUE),
      max = max(values, na.rm = TRUE))
  } else {
    c(min = NA_real_, median = NA_real_, mean = NA_real_, max = NA_real_)
  }
list(summary = summary, values = values)
}

compare_latent_scaling <- function(latents_r, latents_c) {
  L <- min(length(latents_r), length(latents_c))
  if (L == 0) {
    empty_summary <- c(min = NA_real_, median = NA_real_, mean = NA_real_, max = NA_real_)
    return(list(diff_summary = empty_summary,
                ratio_summary = empty_summary,
                diff_values = numeric(0),
                ratio_values = numeric(0),
                meta = data.frame(latent = integer(0), rank_r = integer(0),
                                  rank_c = integer(0), sum_sD2_r = numeric(0),
                                  sum_sD2_c = numeric(0), overlap = integer(0))))
  }
  diff_list <- vector("list", L)
  ratio_list <- vector("list", L)
  meta <- data.frame(
    latent = seq_len(L),
    rank_r = NA_integer_,
    rank_c = NA_integer_,
    sum_sD2_r = NA_real_,
    sum_sD2_c = NA_real_,
    overlap = 0L
  )
  for (l in seq_len(L)) {
    s_r <- latents_r[[l]]$sD2
    s_c <- latents_c[[l]]$sD2
    if (is.null(s_r) || is.null(s_c)) {
      meta$rank_r[l] <- if (!is.null(s_r)) length(s_r) else NA_integer_
      meta$rank_c[l] <- if (!is.null(s_c)) length(s_c) else NA_integer_
      next
    }
    m <- min(length(s_r), length(s_c))
    meta$rank_r[l] <- length(s_r)
    meta$rank_c[l] <- length(s_c)
    meta$sum_sD2_r[l] <- sum(s_r)
    meta$sum_sD2_c[l] <- sum(s_c)
    meta$overlap[l] <- m
    if (m == 0) {
      next
    }
    s_r_sub <- s_r[seq_len(m)]
    s_c_sub <- s_c[seq_len(m)]
    diff_list[[l]] <- s_c_sub - s_r_sub
    ratio_list[[l]] <- ifelse(abs(s_r_sub) > 1e-8, s_c_sub / s_r_sub, NA_real_)
  }
  diff_vals <- unlist(diff_list, use.names = FALSE)
  ratio_vals <- suppressWarnings(as.numeric(unlist(ratio_list, use.names = FALSE)))
  ratio_vals <- ratio_vals[is.finite(ratio_vals)]
  diff_summary <- if (length(diff_vals)) {
    c(min = min(diff_vals),
      median = stats::median(diff_vals),
      mean = mean(diff_vals),
      max = max(diff_vals))
  } else {
    c(min = NA_real_, median = NA_real_, mean = NA_real_, max = NA_real_)
  }
  ratio_summary <- if (length(ratio_vals)) {
    c(min = min(ratio_vals),
      median = stats::median(ratio_vals),
      mean = mean(ratio_vals),
      max = max(ratio_vals))
  } else {
    c(min = NA_real_, median = NA_real_, mean = NA_real_, max = NA_real_)
  }
  list(diff_summary = diff_summary,
       ratio_summary = ratio_summary,
       diff_values = diff_vals,
       ratio_values = ratio_vals,
       meta = meta)
}

compare_implementations <- function(seed = NULL,
                                    n_iter = 20000L,
                                    burnin = 1000L,
                                    thinning = 5L,
                                    inertia_threshold = 0.95,
                                    scale_X = TRUE) {
  set.seed(seed)
  problem <- simulate_problem()

  set.seed(seed)
  time_r <- system.time(
    res_r <- run_brlmm_r(problem$y, problem$X_list, problem$K_list,
                         n_iter = n_iter, burnin = burnin, thinning = thinning,
                         inertia_threshold = inertia_threshold, scale_X = scale_X)
  )

  set.seed(seed)
  time_c <- system.time(
    res_c <- run_brlmm_c(problem$y, problem$X_list, problem$K_list,
                         n_iter = n_iter, burnin = burnin, thinning = thinning,
                         inertia_threshold = inertia_threshold, scale_X = scale_X)
  )

  if (length(res_r$mu_vec) != length(res_c$mu_vec)) {
    stop("Sample sizes differ between implementations", call. = FALSE)
  }

  stats_r <- summarise_result(res_r)
  stats_c <- summarise_result(res_c)
  chain_diff <- diff_summary(stats_r, stats_c)

  y_chain_r <- chain_predict(res_r,
                             new_X_list = problem$X_list,
                             new_K_list = problem$K_list)
  y_chain_c <- res_c$y_chain
  if (!is.matrix(y_chain_c) || !is.matrix(y_chain_r)) {
    stop("Prediction chains missing for comparison", call. = FALSE)
  }
  if (!identical(dim(y_chain_r), dim(y_chain_c))) {
    stop("Prediction chain dimensions differ", call. = FALSE)
  }

  mean_r <- rowMeans(y_chain_r)
  mean_c <- rowMeans(y_chain_c)
  var_r <- apply(y_chain_r, 1, var)
  var_c <- apply(y_chain_c, 1, var)
  diff_mean <- mean_r - mean_c
  avg_var <- (var_r + var_c) / 2
  ratio <- abs(diff_mean) / sqrt(avg_var)
  ratio[!is.finite(ratio)] <- NA_real_

  pred_summary <- list(
    max_abs_mean_diff = max(abs(diff_mean)),
    median_abs_mean_diff = stats::median(abs(diff_mean)),
    max_ratio = max(ratio, na.rm = TRUE),
    median_ratio = stats::median(ratio, na.rm = TRUE),
    mean_var_r = mean(var_r),
    mean_var_c = mean(var_c)
  )
  pred_details <- data.frame(
    individual = seq_along(mean_r),
    mean_r = mean_r,
    mean_c = mean_c,
    mean_diff = diff_mean,
    abs_mean_diff = abs(diff_mean),
    var_r = var_r,
    var_c = var_c,
    ratio = ratio,
    stringsAsFactors = FALSE
  )

  latent_cor <- compute_latent_correlation(res_r$latent_list, res_c$latent_list)
  latent_scale <- compare_latent_scaling(res_r$latent_list, res_c$latent_list)

  list(
    chain = chain_diff,
    prediction = list(summary = pred_summary, details = pred_details),
    runtime = list(
      r_elapsed = unname(time_r["elapsed"]),
      c_elapsed = unname(time_c["elapsed"]),
      speedup = unname(time_r["elapsed"]) / unname(time_c["elapsed"])
    ),
    latent_correlation = latent_cor,
    latent_scaling = latent_scale
  )
}

cat("Running BRLMM comparison test...\n")
summary <- compare_implementations()
cat("\nChain statistics (mean/variance differences):\n")
print(summary$chain)
cat("\nPrediction summary (mean differences vs variances):\n")
print(summary$prediction$summary)
cat("\nPer-individual prediction comparison (first 10 rows):\n")
print(utils::head(summary$prediction$details, 10))
cat("\nLatent U |cor| summary:\n")
print(summary$latent_correlation$summary)
cat("Comparable latent columns:", length(summary$latent_correlation$values),"\n")
cat("Latent sD2 diff summary:\n")
print(summary$latent_scaling$diff_summary)
cat("Latent sD2 ratio summary:\n")
print(summary$latent_scaling$ratio_summary)
cat("Latent rank/overlap info (first 10):\n")
print(stats::na.omit(head(summary$latent_scaling$meta, 10)))
cat("\nRuntime comparison (seconds):\n")
print(summary$runtime)
cat("\nTest complete.\n")
