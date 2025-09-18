rnorm_mat <- function(rows, cols) {
  matrix(rnorm(rows * cols), nrow = rows, ncol = cols)
}

simulation_brlmm <- function(LX=10, LK=10, n_train=100,n_test=100, k=5,r2=0.8,p=20) {
  L = LX + LK
  if(L == 0){
    stop("Need at least one of LX or LK to be non null")
  }
  n = n_train + n_test
  if(n <= 0 || n_train <= 0){
    stop("n and n_train should be positive integers")
  }
  Knull = (LK == 0)
  Xnull = (LX == 0)
  sim = list()
  sim$X_list = NULL
  sim$K_list = NULL
  if(!Xnull){
    sim$X_list = lapply(1:LX,function(l){
      Z = scale(rnorm_mat(n,k))
      Z = sweep(Z,2,rnorm(k),"*")
      Q = rnorm_mat(k,p)
      return(Z%*%Q)
    })
    sim$beta_list = lapply(1:LX,function(l){
      return(rnorm(ncol(sim$X_list[[l]])))
    })
  }
  if(!Knull){
    sim$K_list = lapply(1:LK,function(l){
      Z = scale(rnorm_mat(n,k))
      Z = sweep(Z,2,rnorm(k),"*")
      return(tcrossprod(Z))
    })
    sim$u_list = lapply(sim$K_list,function(K){
      svdK = svd(K)
      svdK$u %*% (svdK$d * rnorm(length(svdK$d)))
    })
  }
  sim$mu = rnorm(1)
  sim$lambda_vec = ((1:L - 1)%%5 == 0)*1
  sim$y = rep(sim$mu,n)
  if(!Xnull){
    sim$y = sim$y + Reduce("+", lapply(1:LX, function(l) {
      if(sim$lambda_vec[l]==0){return(rep(0,n))}
      X = sim$X_list[[l]]
      beta = sim$beta_list[[l]]
      return(X %*% beta)
    }))
    sim$X_new_list = lapply(sim$X_list,function(X){
      X[(n_train+1 ):n,,drop=FALSE]
    })
    sim$X_list = lapply(sim$X_list,function(X){
      X[1:n_train,,drop=FALSE]
    })
  }
  if(!Knull){
    sim$y = sim$y + Reduce("+", lapply(1:LK, function(l) {
      if(sim$lambda_vec[LX+l]==0){return(rep(0,n))}
      return(sim$u_list[[l]])
    }))
    sim$K_new_list = lapply(sim$K_list,function(K){
      K[(n_train+1 ):n,1:n_train,drop=FALSE]
    })
    sim$K_list = lapply(sim$K_list,function(K){
      K[1:n_train,1:n_train,drop=FALSE]
    })
  }
  sim$y = scale(sim$y)*sqrt(r2) + rnorm(n, sd = sqrt(1 - r2))
  sim$y_new = sim$y[(n_train+1 ):n,drop=FALSE]
  sim$y = sim$y[1:n_train,drop=FALSE]
  return(sim)
}
