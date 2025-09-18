#include "brlmm_utils.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <R_ext/RS.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>

int brlmm_copy_row_to_col(const double *src, size_t rows, size_t cols, double **out) {
    if (!src || !out) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    double *dst = (double *)malloc(rows * cols * sizeof(double));
    if (!dst) {
        return BRLMM_ERR_MEMORY;
    }
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    *out = dst;
    return BRLMM_OK;
}

void brlmm_copy_col_to_row(const double *src, size_t rows, size_t cols, double *dst) {
    if (!src || !dst) {
        return;
    }
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            dst[r * cols + c] = src[c * rows + r];
        }
    }
}

int brlmm_allocate_vector(BrlmmVector *vec, size_t size) {
    if (!vec || size == 0) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    vec->data = (double *)calloc(size, sizeof(double));
    if (!vec->data) {
        vec->size = 0;
        return BRLMM_ERR_MEMORY;
    }
    vec->size = size;
    return BRLMM_OK;
}

void brlmm_free_vector(BrlmmVector *vec) {
    if (!vec) {
        return;
    }
    free(vec->data);
    vec->data = NULL;
    vec->size = 0;
}

int brlmm_allocate_matrix(BrlmmMatrix *mat, size_t rows, size_t cols) {
    if (!mat || rows == 0 || cols == 0) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    mat->data = (double *)calloc(rows * cols, sizeof(double));
    if (!mat->data) {
        mat->rows = 0;
        mat->cols = 0;
        return BRLMM_ERR_MEMORY;
    }
    mat->rows = rows;
    mat->cols = cols;
    return BRLMM_OK;
}

void brlmm_free_matrix(BrlmmMatrix *mat) {
    if (!mat) {
        return;
    }
    free(mat->data);
    mat->data = NULL;
    mat->rows = 0;
    mat->cols = 0;
}

void brlmm_vector_copy(double *dst, const double *src, size_t n) {
    if (!dst || !src || n == 0) {
        return;
    }
    memcpy(dst, src, n * sizeof(double));
}

void brlmm_matrix_copy(double *dst, const double *src, size_t n) {
    if (!dst || !src || n == 0) {
        return;
    }
    memcpy(dst, src, n * sizeof(double));
}

void brlmm_vector_zero(double *data, size_t n) {
    if (!data || n == 0) {
        return;
    }
    memset(data, 0, n * sizeof(double));
}

double brlmm_vector_mean(const double *data, size_t n) {
    if (!data || n == 0) {
        return 0.0;
    }
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    return sum / (double)n;
}

double brlmm_vector_variance(const double *data, size_t n) {
    if (!data || n < 2) {
        return 0.0;
    }
    double mean = brlmm_vector_mean(data, n);
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = data[i] - mean;
        sum += diff * diff;
    }
    return sum / (double)(n - 1);
}

double brlmm_vector_sum_sq(const double *data, size_t n) {
    if (!data || n == 0) {
        return 0.0;
    }
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += data[i] * data[i];
    }
    return sum;
}

void brlmm_matrix_vector_mul(const BrlmmMatrix *mat, const double *vec, double *out) {
    if (!mat || !mat->data || !vec || !out) {
        return;
    }
    double *col = NULL;
    if (brlmm_copy_row_to_col(mat->data, mat->rows, mat->cols, &col) != BRLMM_OK) {
        return;
    }
    int m = (int)mat->rows;
    int n = (int)mat->cols;
    int lda = m;
    int inc = 1;
    double alpha = 1.0;
    double beta = 0.0;
    F77_CALL(dgemv)("N", &m, &n, &alpha, col, &lda, vec, &inc, &beta, out, &inc FCONE);
    free(col);
}

void brlmm_matrix_transpose_vector_mul(const BrlmmMatrix *mat, const double *vec, double *out) {
    if (!mat || !mat->data || !vec || !out) {
        return;
    }
    double *col = NULL;
    if (brlmm_copy_row_to_col(mat->data, mat->rows, mat->cols, &col) != BRLMM_OK) {
        return;
    }
    int m = (int)mat->rows;
    int n = (int)mat->cols;
    int lda = m;
    int inc = 1;
    double alpha = 1.0;
    double beta = 0.0;
    F77_CALL(dgemv)("T", &m, &n, &alpha, col, &lda, vec, &inc, &beta, out, &inc FCONE);
    free(col);
}

double brlmm_rng_normal(const BrlmmRng *rng, double mean, double sd) {
    if (!rng || !rng->normal) {
        return mean;
    }
    return rng->normal(rng->state, mean, sd);
}

double brlmm_rng_invgamma(const BrlmmRng *rng, double shape, double rate) {
    if (!rng || !rng->gamma || shape <= 0.0 || rate <= 0.0) {
        return 0.0;
    }
    double sample = rng->gamma(rng->state, shape, rate);
    if (sample <= 0.0) {
        return 0.0;
    }
    return 1.0 / sample;
}

double brlmm_rng_uniform(const BrlmmRng *rng, double min, double max) {
    if (!rng || !rng->uniform) {
        return 0.5 * (min + max);
    }
    return rng->uniform(rng->state, min, max);
}

int brlmm_shuffle_indices(size_t n, size_t *indices, const BrlmmRng *rng) {
    if (!indices) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < n; ++i) {
        indices[i] = i;
    }
    if (!rng || !rng->uniform) {
        return BRLMM_OK;
    }
    for (size_t i = n; i > 1; --i) {
        double u = brlmm_rng_uniform(rng, 0.0, 1.0);
        size_t j = (size_t)floor(u * (double)i);
        if (j >= i) {
            j = i - 1;
        }
        size_t idx_i = i - 1;
        size_t tmp = indices[idx_i];
        indices[idx_i] = indices[j];
        indices[j] = tmp;
    }
    return BRLMM_OK;
}

int brlmm_compute_column_stats(const BrlmmMatrix *mat, double *means, double *stddevs) {
    if (!mat || !mat->data || !means || !stddevs) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    size_t rows = mat->rows;
    size_t cols = mat->cols;
    for (size_t j = 0; j < cols; ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < rows; ++i) {
            sum += mat->data[i * cols + j];
        }
        double mean = sum / (double)rows;
        means[j] = mean;
        double var_acc = 0.0;
        for (size_t i = 0; i < rows; ++i) {
            double diff = mat->data[i * cols + j] - mean;
            var_acc += diff * diff;
        }
        if (rows > 1) {
            var_acc /= (double)(rows - 1);
        } else {
            var_acc = 0.0;
        }
        stddevs[j] = (var_acc > 0.0) ? sqrt(var_acc) : 0.0;
    }
    return BRLMM_OK;
}

void brlmm_apply_column_center_scale(double *data, size_t rows, size_t cols,
                                     const double *means, const double *stddevs) {
    if (!data || !means || !stddevs) {
        return;
    }
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double *cell = data + i * cols + j;
            *cell -= means[j];
            double sd = stddevs[j];
            if (sd != 0.0) {
                *cell /= sd;
            } else {
                *cell = 0.0;
            }
        }
    }
}

int brlmm_compute_xtx(const double *data, size_t rows, size_t cols, double *out) {
    if (!data || !out) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    double *col = NULL;
    if (brlmm_copy_row_to_col(data, rows, cols, &col) != BRLMM_OK) {
        return BRLMM_ERR_MEMORY;
    }
    double *colC = (double *)calloc(cols * cols, sizeof(double));
    if (!colC) {
        free(col);
        return BRLMM_ERR_MEMORY;
    }
    int m = (int)cols;
    int n = (int)cols;
    int k = (int)rows;
    int lda = (int)rows;
    int ldb = (int)rows;
    int ldc = m;
    double alpha = 1.0;
    double beta = 0.0;
    F77_CALL(dgemm)("T", "N", &m, &n, &k, &alpha, col, &lda, col, &ldb, &beta, colC, &ldc FCONE FCONE);
    brlmm_copy_col_to_row(colC, cols, cols, out);
    free(colC);
    free(col);
    return BRLMM_OK;
}

void brlmm_matrix_multiply_raw(const double *A, size_t a_rows, size_t a_cols,
                               const double *B, size_t b_cols, double *out) {
    if (!A || !B || !out) {
        return;
    }
    double *colA = NULL;
    double *colB = NULL;
    if (brlmm_copy_row_to_col(A, a_rows, a_cols, &colA) != BRLMM_OK ||
        brlmm_copy_row_to_col(B, a_cols, b_cols, &colB) != BRLMM_OK) {
        free(colA);
        free(colB);
        return;
    }
    double *colC = (double *)calloc(a_rows * b_cols, sizeof(double));
    if (!colC) {
        free(colA);
        free(colB);
        return;
    }
    int m = (int)a_rows;
    int n = (int)b_cols;
    int k = (int)a_cols;
    int lda = (int)a_rows;
    int ldb = (int)a_cols;
    int ldc = m;
    double alpha = 1.0;
    double beta = 0.0;
    F77_CALL(dgemm)("N", "N", &m, &n, &k, &alpha, colA, &lda, colB, &ldb, &beta, colC, &ldc FCONE FCONE);
    brlmm_copy_col_to_row(colC, a_rows, b_cols, out);
    free(colC);
    free(colA);
    free(colB);
}
