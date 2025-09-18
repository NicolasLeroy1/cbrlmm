#include "brlmm.h"
#include "brlmm_utils.h"

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <Rmath.h>

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

static double r_rng_normal(void *state, double mean, double sd) {
    (void)state;
    return Rf_rnorm(mean, sd);
}

static double r_rng_gamma(void *state, double shape, double rate) {
    (void)state;
    if (rate <= 0.0) {
        return 0.0;
    }
    double scale = 1.0 / rate;
    return Rf_rgamma(shape, scale);
}

static double r_rng_uniform(void *state, double min, double max) {
    (void)state;
    return Rf_runif(min, max);
}

static int extract_matrix(SEXP mat, BrlmmMatrix *out, double **owner) {
    if (!Rf_isReal(mat)) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    SEXP dims = Rf_getAttrib(mat, R_DimSymbol);
    if (Rf_length(dims) != 2) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    size_t rows = (size_t)INTEGER(dims)[0];
    size_t cols = (size_t)INTEGER(dims)[1];
    double *src = REAL(mat);
    double *buffer = (double *)malloc(rows * cols * sizeof(double));
    if (!buffer) {
        return BRLMM_ERR_MEMORY;
    }
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            buffer[r * cols + c] = src[r + rows * c];
        }
    }
    out->rows = rows;
    out->cols = cols;
    out->data = buffer;
    if (owner) {
        *owner = buffer;
    }
    return BRLMM_OK;
}

static void copy_row_major_to_matrix(SEXP mat, const double *data, size_t rows, size_t cols) {
    double *dst = REAL(mat);
    for (size_t c = 0; c < cols; ++c) {
        for (size_t r = 0; r < rows; ++r) {
            dst[c * rows + r] = data[r * cols + c];
        }
    }
}

static SEXP build_result_list(const BrlmmOutput *output,
                              const BrlmmMatrix *pred_chain) {
    size_t sample_count = output->sample_count;
    size_t L = output->L;
    size_t n = output->n;

    SEXP result = PROTECT(Rf_allocVector(VECSXP, 11));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 11));

    // mu_vec --------------------------------------------------------------
    SEXP mu_vec = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)sample_count));
    if (sample_count > 0) {
        memcpy(REAL(mu_vec), output->mu_chain.data, sample_count * sizeof(double));
    }

    // sigma2_vec ---------------------------------------------------------
    SEXP sigma2_vec = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)sample_count));
    if (sample_count > 0) {
        memcpy(REAL(sigma2_vec), output->sigma2_chain.data, sample_count * sizeof(double));
    }

    // mean_squared_error_vec ---------------------------------------------
    SEXP mse_vec = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)sample_count));
    if (sample_count > 0) {
        memcpy(REAL(mse_vec), output->mse_chain.data, sample_count * sizeof(double));
    }

    // loss_vector --------------------------------------------------------
    SEXP loss_vec = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)sample_count));
    if (sample_count > 0) {
        memcpy(REAL(loss_vec), output->loss_chain.data, sample_count * sizeof(double));
    }

    // lambda_frame -------------------------------------------------------
    SEXP lambda_mat = PROTECT(Rf_allocMatrix(REALSXP, (int)L, (int)sample_count));
    double *lambda_dst = REAL(lambda_mat);
    const double *lambda_src = output->lambda_chain.data;
    for (size_t col = 0; col < sample_count; ++col) {
        for (size_t row = 0; row < L; ++row) {
            lambda_dst[col * L + row] = lambda_src[row * sample_count + col];
        }
    }

    // nu_frame_list ------------------------------------------------------
    SEXP nu_list = PROTECT(Rf_allocVector(VECSXP, (R_xlen_t)L));
    for (size_t l = 0; l < L; ++l) {
        size_t entry_size = output->nu_chain.items[l].size;
        size_t rank = (sample_count > 0) ? entry_size / sample_count : 0;
        SEXP nu_mat = PROTECT(Rf_allocMatrix(REALSXP, (int)rank, (int)sample_count));
        double *nu_dst = REAL(nu_mat);
        const double *nu_src = output->nu_chain.items[l].data;
        for (size_t col = 0; col < sample_count; ++col) {
            for (size_t row = 0; row < rank; ++row) {
                nu_dst[col * rank + row] = nu_src[row * sample_count + col];
            }
        }
        SET_VECTOR_ELT(nu_list, (R_xlen_t)l, nu_mat);
        UNPROTECT(1); // nu_mat
    }

    // prediction chain ---------------------------------------------------
    SEXP pred_mat = PROTECT(Rf_allocMatrix(REALSXP, (int)n, (int)sample_count));
    double *pred_dst = REAL(pred_mat);
    size_t pred_len = (size_t)n * sample_count;
    if (pred_len > 0) {
        if (pred_chain && pred_chain->data) {
            const double *pred_src = pred_chain->data;
            for (size_t col = 0; col < sample_count; ++col) {
                for (size_t row = 0; row < n; ++row) {
                    pred_dst[row + n * col] = pred_src[row * sample_count + col];
                }
            }
        } else {
            for (size_t i = 0; i < pred_len; ++i) {
                pred_dst[i] = NA_REAL;
            }
        }
    }

    // mu_mean ------------------------------------------------------------
    SEXP mu_mean = PROTECT(Rf_ScalarReal(output->mu_mean));

    // metadata list ------------------------------------------------------
    SEXP meta = PROTECT(Rf_allocVector(VECSXP, 5));
    SEXP meta_names = PROTECT(Rf_allocVector(STRSXP, 5));
    SET_VECTOR_ELT(meta, 0, Rf_ScalarInteger((int)output->sample_count));
    SET_STRING_ELT(meta_names, 0, Rf_mkChar("sample_count"));
    SET_VECTOR_ELT(meta, 1, Rf_ScalarInteger((int)output->L));
    SET_STRING_ELT(meta_names, 1, Rf_mkChar("L"));
    SET_VECTOR_ELT(meta, 2, Rf_ScalarInteger((int)output->LX));
    SET_STRING_ELT(meta_names, 2, Rf_mkChar("LX"));
    SET_VECTOR_ELT(meta, 3, Rf_ScalarInteger((int)output->LK));
    SET_STRING_ELT(meta_names, 3, Rf_mkChar("LK"));
    SET_VECTOR_ELT(meta, 4, Rf_ScalarInteger((int)output->n));
    SET_STRING_ELT(meta_names, 4, Rf_mkChar("n"));
    Rf_setAttrib(meta, R_NamesSymbol, meta_names);

    SEXP latent_out = PROTECT(Rf_allocVector(VECSXP, (R_xlen_t)L));
    for (size_t l = 0; l < L; ++l) {
        const BrlmmLatent *latent = &output->latents.items[l];
        SEXP latent_obj = PROTECT(Rf_allocVector(VECSXP, 9));
        SEXP latent_names = PROTECT(Rf_allocVector(STRSXP, 9));
        int field = 0;

        SET_VECTOR_ELT(latent_obj, field, Rf_ScalarInteger((int)latent->kind));
        SET_STRING_ELT(latent_names, field++, Rf_mkChar("kind"));

        SET_VECTOR_ELT(latent_obj, field, Rf_ScalarInteger((int)latent->rank));
        SET_STRING_ELT(latent_names, field++, Rf_mkChar("rank"));

        if (latent->U.data && latent->U.rows && latent->U.cols) {
            SEXP U = PROTECT(Rf_allocMatrix(REALSXP, (int)latent->U.rows, (int)latent->U.cols));
            copy_row_major_to_matrix(U, latent->U.data, latent->U.rows, latent->U.cols);
            SET_VECTOR_ELT(latent_obj, field, U);
            UNPROTECT(1);
        } else {
            SET_VECTOR_ELT(latent_obj, field, R_NilValue);
        }
        SET_STRING_ELT(latent_names, field++, Rf_mkChar("U"));

        if (latent->V.data && latent->V.rows && latent->V.cols) {
            SEXP V = PROTECT(Rf_allocMatrix(REALSXP, (int)latent->V.rows, (int)latent->V.cols));
            copy_row_major_to_matrix(V, latent->V.data, latent->V.rows, latent->V.cols);
            SET_VECTOR_ELT(latent_obj, field, V);
            UNPROTECT(1);
        } else {
            SET_VECTOR_ELT(latent_obj, field, R_NilValue);
        }
        SET_STRING_ELT(latent_names, field++, Rf_mkChar("V"));

        if (latent->scaled.data && latent->scaled.size) {
            SEXP sD2 = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)latent->scaled.size));
            brlmm_vector_copy(REAL(sD2), latent->scaled.data, latent->scaled.size);
            SET_VECTOR_ELT(latent_obj, field, sD2);
            UNPROTECT(1);
        } else {
            SET_VECTOR_ELT(latent_obj, field, R_NilValue);
        }
        SET_STRING_ELT(latent_names, field++, Rf_mkChar("sD2"));

        if (latent->singular.data && latent->singular.size) {
            SEXP D = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)latent->singular.size));
            brlmm_vector_copy(REAL(D), latent->singular.data, latent->singular.size);
            SET_VECTOR_ELT(latent_obj, field, D);
            UNPROTECT(1);
        } else {
            SET_VECTOR_ELT(latent_obj, field, R_NilValue);
        }
        SET_STRING_ELT(latent_names, field++, Rf_mkChar("D"));

        if (latent->eigen.data && latent->eigen.size) {
            SEXP D2 = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)latent->eigen.size));
            brlmm_vector_copy(REAL(D2), latent->eigen.data, latent->eigen.size);
            SET_VECTOR_ELT(latent_obj, field, D2);
            UNPROTECT(1);
        } else {
            SET_VECTOR_ELT(latent_obj, field, R_NilValue);
        }
        SET_STRING_ELT(latent_names, field++, Rf_mkChar("D2"));

        if (latent->centers.data && latent->centers.size) {
            SEXP centers = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)latent->centers.size));
            brlmm_vector_copy(REAL(centers), latent->centers.data, latent->centers.size);
            SET_VECTOR_ELT(latent_obj, field, centers);
            UNPROTECT(1);
        } else {
            SET_VECTOR_ELT(latent_obj, field, R_NilValue);
        }
        SET_STRING_ELT(latent_names, field++, Rf_mkChar("centers"));

        if (latent->scale.data && latent->scale.size) {
            SEXP scale = PROTECT(Rf_allocVector(REALSXP, (R_xlen_t)latent->scale.size));
            brlmm_vector_copy(REAL(scale), latent->scale.data, latent->scale.size);
            SET_VECTOR_ELT(latent_obj, field, scale);
            UNPROTECT(1);
        } else {
            SET_VECTOR_ELT(latent_obj, field, R_NilValue);
        }
        SET_STRING_ELT(latent_names, field++, Rf_mkChar("scale"));

        Rf_setAttrib(latent_obj, R_NamesSymbol, latent_names);
        SET_VECTOR_ELT(latent_out, (R_xlen_t)l, latent_obj);
        UNPROTECT(2); // latent_obj, latent_names
    }

    SET_VECTOR_ELT(result, 0, mu_vec);
    SET_STRING_ELT(names, 0, Rf_mkChar("mu_vec"));

    SET_VECTOR_ELT(result, 1, sigma2_vec);
    SET_STRING_ELT(names, 1, Rf_mkChar("sigma2_vec"));

    SET_VECTOR_ELT(result, 2, lambda_mat);
    SET_STRING_ELT(names, 2, Rf_mkChar("lambda_frame"));

    SET_VECTOR_ELT(result, 3, nu_list);
    SET_STRING_ELT(names, 3, Rf_mkChar("nu_frame_list"));

    SET_VECTOR_ELT(result, 4, pred_mat);
    SET_STRING_ELT(names, 4, Rf_mkChar("y_chain"));

    SET_VECTOR_ELT(result, 5, mse_vec);
    SET_STRING_ELT(names, 5, Rf_mkChar("mean_squared_error_vec"));

    SET_VECTOR_ELT(result, 6, loss_vec);
    SET_STRING_ELT(names, 6, Rf_mkChar("loss_vector"));

    SET_VECTOR_ELT(result, 7, mu_mean);
    SET_STRING_ELT(names, 7, Rf_mkChar("mu_mean"));

    SET_VECTOR_ELT(result, 8, meta);
    SET_STRING_ELT(names, 8, Rf_mkChar("metadata"));

    SET_VECTOR_ELT(result, 9, latent_out);
    SET_STRING_ELT(names, 9, Rf_mkChar("latent_list"));

    SET_VECTOR_ELT(result, 10, Rf_ScalarLogical(1));
    SET_STRING_ELT(names, 10, Rf_mkChar("from_c"));

    Rf_setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(13); // result,names,mu_vec,sigma2_vec,mse_vec,loss_vec,lambda_mat,nu_list,pred_mat,mu_mean,meta,meta_names,latent_out
    return result;
}

SEXP C_brlmm_run(SEXP y,
                 SEXP X_list,
                 SEXP K_list,
                 SEXP n_iter,
                 SEXP burnin,
                 SEXP thinning,
                 SEXP inertia_threshold,
                 SEXP scale_X) {
    if (!Rf_isReal(y)) {
        Rf_error("y must be a numeric vector");
    }

    BrlmmProblem problem;
    memset(&problem, 0, sizeof(problem));
    problem.y.data = REAL(y);
    problem.y.size = (size_t)Rf_length(y);

    BrlmmMatrix *X_items = NULL;
    double **X_buffers = NULL;
    BrlmmMatrix *K_items = NULL;
    double **K_buffers = NULL;
    size_t X_count = 0;
    size_t K_count = 0;

    if (!Rf_isNull(X_list)) {
        if (!Rf_isNewList(X_list)) {
            Rf_error("X_list must be a list of matrices");
        }
        R_xlen_t LX = Rf_length(X_list);
        X_items = (BrlmmMatrix *)calloc((size_t)LX, sizeof(BrlmmMatrix));
        X_buffers = (double **)calloc((size_t)LX, sizeof(double *));
        if (!X_items || !X_buffers) {
            free(X_items);
            free(X_buffers);
            Rf_error("memory allocation failure for X_list");
        }
        for (R_xlen_t i = 0; i < LX; ++i) {
            int rc = extract_matrix(VECTOR_ELT(X_list, i), &X_items[i], &X_buffers[i]);
            if (rc != BRLMM_OK) {
                for (R_xlen_t j = 0; j < i; ++j) {
                    free(X_buffers[j]);
                }
                free(X_buffers);
                free(X_items);
                Rf_error("invalid matrix in X_list");
            }
        }
        problem.X_list.items = X_items;
        problem.X_list.count = (size_t)LX;
        X_count = (size_t)LX;
    }

    if (!Rf_isNull(K_list)) {
        if (!Rf_isNewList(K_list)) {
            if (X_buffers) {
                for (size_t j = 0; j < X_count; ++j) {
                    free(X_buffers[j]);
                }
            }
            free(X_buffers);
            free(X_items);
            Rf_error("K_list must be a list of matrices");
        }
        R_xlen_t LK = Rf_length(K_list);
        K_items = (BrlmmMatrix *)calloc((size_t)LK, sizeof(BrlmmMatrix));
        K_buffers = (double **)calloc((size_t)LK, sizeof(double *));
        if (!K_items || !K_buffers) {
            if (X_buffers) {
                for (size_t j = 0; j < X_count; ++j) {
                    free(X_buffers[j]);
                }
            }
            free(X_buffers);
            free(X_items);
            free(K_items);
            free(K_buffers);
            Rf_error("memory allocation failure for K_list");
        }
        for (R_xlen_t i = 0; i < LK; ++i) {
            int rc = extract_matrix(VECTOR_ELT(K_list, i), &K_items[i], &K_buffers[i]);
            if (rc != BRLMM_OK) {
                if (X_buffers) {
                    for (size_t j = 0; j < X_count; ++j) {
                        free(X_buffers[j]);
                    }
                }
                for (R_xlen_t j = 0; j < i; ++j) {
                    free(K_buffers[j]);
                }
                free(X_buffers);
                free(X_items);
                free(K_buffers);
                free(K_items);
                Rf_error("invalid matrix in K_list");
            }
        }
        problem.K_list.items = K_items;
        problem.K_list.count = (size_t)LK;
        K_count = (size_t)LK;
    }

    BrlmmConfig config;
    config.n_iter = (size_t)Rf_asInteger(n_iter);
    config.burnin = (size_t)Rf_asInteger(burnin);
    config.thinning = (size_t)Rf_asInteger(thinning);
    config.inertia_threshold = Rf_asReal(inertia_threshold);
    config.scale_X = Rf_asLogical(scale_X) ? true : false;
    config.method = BRLMM_METHOD_HORSESHOE;

    BrlmmRng rng;
    rng.normal = r_rng_normal;
    rng.gamma = r_rng_gamma;
    rng.uniform = r_rng_uniform;
    rng.state = NULL;

    BrlmmOutput output;
    memset(&output, 0, sizeof(output));

    GetRNGstate();
    int rc = brlmm_run(&problem, &config, &rng, &output);
    PutRNGstate();

    BrlmmMatrix pred_chain;
    memset(&pred_chain, 0, sizeof(pred_chain));
    bool have_pred = false;

    if (rc == BRLMM_OK && output.sample_count > 0 && output.n > 0) {
        const BrlmmMatrixList *new_X = (problem.X_list.count > 0) ? &problem.X_list : NULL;
        const BrlmmMatrixList *new_K = (problem.K_list.count > 0) ? &problem.K_list : NULL;
        if (new_X || new_K) {
            if (brlmm_allocate_matrix(&pred_chain, output.n, output.sample_count) == BRLMM_OK) {
                int pred_rc = brlmm_predict(&output, new_X, new_K, &pred_chain);
                if (pred_rc == BRLMM_OK) {
                    have_pred = true;
                } else {
                    brlmm_free_matrix(&pred_chain);
                }
            }
        }
    }

    if (X_buffers) {
        for (size_t i = 0; i < X_count; ++i) {
            free(X_buffers[i]);
        }
    }
    if (K_buffers) {
        for (size_t i = 0; i < K_count; ++i) {
            free(K_buffers[i]);
        }
    }
    free(X_buffers);
    free(K_buffers);
    free(X_items);
    free(K_items);

    if (rc != BRLMM_OK) {
        brlmm_output_clear(&output);
        Rf_error("brlmm_run failed with code %d", rc);
    }

    SEXP result = PROTECT(build_result_list(&output, have_pred ? &pred_chain : NULL));
    if (have_pred) {
        brlmm_free_matrix(&pred_chain);
    }
    brlmm_output_clear(&output);
    UNPROTECT(1);
    return result;
}

static const R_CallMethodDef CallEntries[] = {
    {"C_brlmm_run", (DL_FUNC)&C_brlmm_run, 8},
    {NULL, NULL, 0}
};

void R_init_brlmm_port(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
