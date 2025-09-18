
brlm = function(y,
                X_list = NULL,
                K_list = NULL,
                niter = 5000,
                burnin = 1000,
                thinning = 5,
                inertia_threshold = 0.95,
                method = "horseshoe",
                X_matrix_scaling=TRUE) {
  input = parse_input(y=y,
                      X_list=X_list,
                      K_list=K_list,
                      method=method,
                      niter=niter,
                      burnin=burnin,
                      thinning=thinning,
                      inertia_threshold=inertia_threshold,
                      X_matrix_scaling=X_matrix_scaling)
  chain = init_chain(input, method)
  current = init_current(input, method)
  for (iter in 1:niter) {
    if(method == "horseshoe"){
      current = update_current_horseshoe(current, input$latent_list, input$L, input$n)
    } else {
      stop("undefined method")
    }
    s = iter - burnin
    if (s > 0) {
      if (s %% thinning == 0) {
        s = s / thinning
        chain[[s]] = update_chain(input, current, method)
      }
    }
  }
  return(make_output(input, chain, method))
}

parse_input = function(y,
                       X_list,
                       K_list,
                       method,
                       niter,
                       burnin,
                       thinning,
                       inertia_threshold,
                       X_matrix_scaling=TRUE) {
  methods = c("horseshoe")
  Knull = is.null(K_list) || length(K_list)==0
  Xnull = is.null(X_list) || length(X_list)==0
  LX = 0
  LK = 0
  n = length(y)
  if (is.null(y) || n== 0) {
    stop("y should be a non-empty vector")
  }
  if (is.matrix(K_list)) {
    K_list = list(K_list)
  }
  if (is.matrix(X_list)) {
    X_list = list(X_list)
  }
  if (Xnull &&
      Knull) {
    stop("Provide a matrix or list of matrices for fixed effects (X) or/and random effects (K)")
  }

  if (niter <= 0 || burnin < 0 || thinning <= 0) {
    stop("niter, burnin and thinning should be positive integers")
  }
  if (niter <= burnin) {
    stop("niter should be greater than burnin")
  }
  if (thinning > niter - burnin) {
    stop("thinning should be less than niter - burnin")
  }
  latent_list = list()
  if (!Xnull) {
    LX = length(X_list)
    for(l in 1:LX) {if(nrow(X_list[[l]])!=n){stop("X_list[[",l,"]] should have the same number of rows as y")}}
    latent_list = c(latent_list, compute_X_latent_list(X_list,inertia_threshold,X_matrix_scaling=X_matrix_scaling))
  }
  if (!Knull) {
    LK = length(K_list)
    for(l in 1:LK) {if(nrow(K_list[[l]])!=n){stop("K_list[[",l,"]] should have the same number of rows as y")}}
    latent_list = c(latent_list, compute_K_latent_list(K_list,inertia_threshold))
  }
  if (!method %in% methods) {
    stop("Method should be in ", methods)
  }
  return(list(
    n = length(y),
    y = y,
    latent_list = latent_list,
    LX = LX,
    LK = LK,
    L = LX + LK
  ))
}

init_current = function(input, method) {
  if (method != "horseshoe") {
    stop("undefined method")
  }
  current = list()
  current$mu = mean(input$y)
  current$residuals = input$y - current$mu
  current$sigma2 = var(input$y)
  current$nu_list = lapply(input$latent_list, function(latent) {
    rep(0, latent$rank)
  })
  current$lambda_vec = rep(1, input$L)
  current$omega2_vec = rep(1, input$L)
  current$tau2 = 1
  current$xi2_omega2_vec = rep(1, input$L)
  current$xi2_tau2 = 1
  return(current)
}

init_chain = function(input, method) {
  chain = list()
  return(chain)
}

update_chain = function(input, current, method) {
  sample=list()
  sample$mu = current$mu
  sample$nu_list = current$nu_list
  sample$sigma2 = current$sigma2
  sample$lambda_vec = current$lambda_vec
  sample$residuals = current$residuals
  sample$loss = compute_loss(input, current, method)
  return(sample)
}

make_output = function(input, chain, method) {
  output = list()
  output$mu_vec = sapply(chain, function(sample) {
    sample$mu
  })
  output$sigma2_vec = sapply(chain, function(sample) {
    sample$sigma2
  })
  output$lambda_frame = matrix(sapply(chain, function(sample) {
    sample$lambda_vec
  }), ncol = length(chain))
  output$nu_frame_list = lapply(1:input$L, function(l) {
    x = sapply(chain, function(sample) {
      sample$nu_list[[l]]
    })
    matrix(x,ncol=length(chain))
  })
  output$explained_variance_vec_list = lapply(1:input$L, function(l) {
    sapply(chain, function(sample) {
      sum(sample$nu_list[[l]]^2) * sample$lambda_vec[l]^2 / input$n
    })
  })
  output$mean_squared_error_vec = sapply(chain, function(sample) {
    mean(sample$residuals^2)
  })
  output$loss_vector = sapply(chain, function(sample) {
    sample$loss
  })
  output$latent_list = input$latent_list
  output$effect_mean_list = lapply(1:input$L, function(l) {
    n_sample = length(chain)
    nu_frame = output$nu_frame_list[[l]]
    lambda_vec = output$lambda_frame[l, ]
    return(nu_frame %*% lambda_vec / n_sample)
  })
  output$mu_mean = mean(output$mu_vec)
  output$LX = input$LX
  output$LK = input$LK
  output$n = input$n
  return(output)
}

compute_X_latent_list = function(X_list,inertia_threshold,X_matrix_scaling=TRUE) {
  return(lapply(X_list, function(X) {
    if(X_matrix_scaling){
      X = scale(X)
      centers = attr(X, "scaled:center")
      sd = attr(X, "scaled:scale")
      X[is.na(X)] = 0
    } else {
      centers = rep(0, ncol(X))
      sd = rep(1, ncol(X))
    }
    temp = tryCatch({svd(X)},
                    error = function(e) {
                      rsvd::rsvd(X, k = max(floor(min(ncol(X), nrow(X))-10),1))
                    })
    sD2 = temp$d^2 / sum(temp$d^2)
    rank = which(cumsum(sD2)>inertia_threshold)[1]
    sD2 = sD2[1:rank]
    sD2 = nrow(X) * sD2/sum(sD2)
    return(list(
      U = temp$u[, 1:rank, drop = FALSE],
      D = temp$d[1:rank],
      centers = centers,
      sd = sd,
      sD2 = sD2,
      V = as.matrix(temp$v[, 1:rank, drop = FALSE]),
      rank=rank
    ))
  }))
}
compute_K_latent_list = function(K_list,inertia_threshold) {
  return(lapply(K_list, function(K) {
    temp = eigen(K)
    sD2 <- temp$values / sum(temp$values)
    rank = which(cumsum(sD2)>inertia_threshold)[1]
    sD2 <- sD2[1:rank]
    sD2 = nrow(K)* sD2/sum(sD2)
    return(list(
      U = temp$vectors[, 1:rank, drop = FALSE],
      sD2 = sD2,
      D2 = temp$values[1:rank],
      rank=rank
    ))
  }))
}

rinv_gamma <- function(a, b){1 / rgamma(1, shape = a, rate = b)}

update_current_horseshoe <- function(current, latent_list, L, n){
  # --- pull copies ------------------------------------------------
  residuals       <- current$residuals
  mu              <- current$mu
  sigma2          <- current$sigma2
  lambda_vec      <- current$lambda_vec
  omega2_vec      <- current$omega2_vec
  tau2            <- current$tau2
  xi2_omega2_vec  <- current$xi2_omega2_vec
  xi2_tau2        <- current$xi2_tau2
  nu_list         <- current$nu_list

  residuals <- residuals + mu
  mu        <- rnorm(1, mean(residuals), sqrt(sigma2 / n))
  residuals <- residuals - mu

  # --------------- factor-by-factor loop --------------------------
  for (l in sample.int(L)) {
    latent   <- latent_list[[l]]
    U        <- latent$U
    sD2      <- latent$sD2
    nu_l     <- nu_list[[l]]
    lambda_l <- lambda_vec[l]
    rnk      <- length(nu_l)
    residuals <- residuals + lambda_l * (U %*% nu_l)
    UtR         <- as.vector(crossprod(U, residuals))
    scale2_nu   <- 1 / (1 / sD2 + lambda_l^2 / sigma2)
    center_nu   <- lambda_l * scale2_nu * UtR / sigma2
    nu_l        <- center_nu + sqrt(scale2_nu) * rnorm(rnk)
    scale2_lam  <- 1 / (sum(nu_l^2) / sigma2 + 1 / (sigma2 * tau2 * omega2_vec[l]))
    center_lam  <- scale2_lam * sum(UtR * nu_l) / sigma2
    lambda_l    <- rnorm(1, center_lam, sqrt(scale2_lam))
    b_omega2       <- lambda_l^2 / (2 * sigma2 * tau2) +
      1 / xi2_omega2_vec[l]
    omega2_vec[l]  <- rinv_gamma(1, b_omega2)
    b_xi2             <- 1 + 1 / omega2_vec[l]
    xi2_omega2_vec[l] <- rinv_gamma(1, b_xi2)
    residuals        <- residuals - lambda_l * (U %*% nu_l)

    lambda_vec[l]  <- lambda_l
    nu_list[[l]]   <- nu_l
  }
  a_tau2  <- 0.5 * (L + 1)
  b_tau2  <- sum(lambda_vec^2 / omega2_vec) / (2 * sigma2) + 1 / xi2_tau2
  tau2    <- rinv_gamma(a_tau2, b_tau2)

  b_xi2_tau2 <- 1 + 1 / tau2
  xi2_tau2   <- rinv_gamma(1, b_xi2_tau2)

  a_sig2 <- 0.5 * (n + L)
  b_sig2 <- sum(residuals^2) / 2 +
    sum(lambda_vec^2 / omega2_vec) / (2 * tau2)
  sigma2 <- rinv_gamma(a_sig2, b_sig2)

  # --- push back ------------------------------------------------
  current$residuals  <- residuals
  current$lambda_vec <- lambda_vec
  current$tau2       <- tau2
  current$xi2_tau2   <- xi2_tau2
  current$sigma2     <- sigma2
  current$mu         <- mu
  current$nu_list    <- nu_list
  current$omega2_vec <- omega2_vec
  current$xi2_omega2_vec <- xi2_omega2_vec
  return(current)
}


chain_predict = function(result,
                         new_X_list = NULL,
                         new_K_list = NULL) {
  Xnull = is.null(new_X_list)
  Knull = is.null(new_K_list)
  if (Knull && Xnull) {
    stop(
      "Provide a list of matrices for fixed effects (new_X_list) or random effects (new_K_list)"
    )
  }
  if (is.matrix(new_X_list)) {
    new_X_list = list(new_X_list)
  }
  if (is.matrix(new_K_list)) {
    new_K_list = list(new_K_list)
  }
  LX = 0
  LK = 0
  n = 0
  new_U_list = list()
  if (!Xnull) {
    LX = length(new_X_list)
    if (LX != result$LX) {
      stop("new_X_list should have the same length as the original X_list")
    }
    n = nrow(new_X_list[[1]])
    temp_new_U_list = lapply(1:LX, function(l) {
      latent = result$latent_list[[l]]
      X = new_X_list[[l]]
      if (nrow(latent$V) != ncol(X)) {
        stop(
          "new_X_list[[",
          l,
          "]] should have the same number of columns as the original X_list[[",
          l,
          "]]"
        )
      }
      X = sweep(as.matrix(X), 2, latent$centers, "-")
      for (j in 1:ncol(X)) {
        if (latent$sd[j] == 0) {
          X[, j] = X[, j] * 0
        } else {
          X[, j] = X[, j] / latent$sd[j]
        }
      }
      new_U = sweep(X %*% latent$V, 2, latent$D, "/")
      return(new_U)
    })
    new_U_list = c(new_U_list, temp_new_U_list)
  }
  if (!Knull) {
    LK = length(new_K_list)
    if (LK != result$LK) {
      stop("new_K_list should have the same length as the original K_list")
    }
    n = nrow(new_K_list[[1]])
    temp_new_U_list = lapply(1:LK, function(l) {
      latent = result$latent_list[[LX + l]]
      new_K = as.matrix(new_K_list[[l]])
      if (result$n != ncol(new_K)) {
        stop(
          "new_K_list[[",
          l,
          "]] should have the same number of columns as the original K_list[[",
          l,
          "]]"
        )
      }
      new_U = sweep(new_K %*% latent$U, 2, latent$D2, "/")
      return(new_U)
    })
    new_U_list = c(new_U_list, temp_new_U_list)
  }
  n_sample = length(result$mu_vec)
  y_chain = matrix(0, nrow = n, ncol = n_sample)
  for (s in 1:n_sample) {
    y = rep(result$mu_vec[s], n)
    if (!Xnull) {
      for (l in 1:LX) {
        effect = result$nu_frame_list[[l]][, s,drop=FALSE] * result$lambda_frame[l, s]
        y = y + new_U_list[[l]] %*% effect
      }
    }
    if (!Knull) {
      for (l in 1:LK) {
        effect = result$nu_frame_list[[LX + l]][, s,drop=FALSE] * result$lambda_frame[LX + l, s]
        y = y + new_U_list[[LX + l]] %*% effect
      }
    }
    y_chain[, s] = y
  }
  return(y_chain)
}


compute_loss = function(input, current, method) {
  if (method != "horseshoe") {
    stop("undefined method")
  }
  latent_indices = seq_len(input$L)
  latent_penalty = sum(sapply(latent_indices, function(l) {
    sD2 = input$latent_list[[l]]$sD2
    sum(current$nu_list[[l]]^2 / sD2) / 2
  }))
  residual_term = sum(current$residuals^2) / (2 * current$sigma2)
  lambda_penalty = sum(current$lambda_vec^2 / current$omega2_vec) /
    (2 * current$tau2 * current$sigma2)
  omega_penalty = sum(log(current$omega2_vec)) * 2
  tau_log_term = (input$L + 3) * log(current$tau2) / 2
  sigma_log_term = (input$n + input$L + 2) * log(current$sigma2) / 2
  tau_xi_terms = 1 / (current$tau2 * current$xi2_tau2) + 1 / current$xi2_tau2 + 2 * log(current$xi2_tau2)
  omega_xi_terms = sum(1 / (current$omega2_vec * current$xi2_omega2_vec)) +
    sum(1 / current$xi2_omega2_vec) + 2 * sum(log(current$xi2_omega2_vec))
  residual_term + latent_penalty + lambda_penalty + omega_penalty +
    tau_log_term + sigma_log_term + tau_xi_terms + omega_xi_terms
}

