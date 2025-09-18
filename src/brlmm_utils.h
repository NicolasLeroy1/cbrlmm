#ifndef BRLMM_UTILS_H
#define BRLMM_UTILS_H

#include <stddef.h>
#include "brlmm.h"

int brlmm_allocate_vector(BrlmmVector *vec, size_t size);
void brlmm_free_vector(BrlmmVector *vec);
int brlmm_allocate_matrix(BrlmmMatrix *mat, size_t rows, size_t cols);
void brlmm_free_matrix(BrlmmMatrix *mat);
void brlmm_vector_copy(double *dst, const double *src, size_t n);
void brlmm_matrix_copy(double *dst, const double *src, size_t n);
void brlmm_vector_zero(double *data, size_t n);
double brlmm_vector_mean(const double *data, size_t n);
double brlmm_vector_variance(const double *data, size_t n);
double brlmm_vector_sum_sq(const double *data, size_t n);
int brlmm_copy_row_to_col(const double *src, size_t rows, size_t cols, double **out);
void brlmm_copy_col_to_row(const double *src, size_t rows, size_t cols, double *dst);
void brlmm_matrix_vector_mul(const BrlmmMatrix *mat, const double *vec, double *out);
void brlmm_matrix_transpose_vector_mul(const BrlmmMatrix *mat, const double *vec, double *out);
double brlmm_rng_normal(const BrlmmRng *rng, double mean, double sd);
double brlmm_rng_invgamma(const BrlmmRng *rng, double shape, double rate);
double brlmm_rng_uniform(const BrlmmRng *rng, double min, double max);
int brlmm_shuffle_indices(size_t n, size_t *indices, const BrlmmRng *rng);
int brlmm_compute_column_stats(const BrlmmMatrix *mat, double *means, double *stddevs);
void brlmm_apply_column_center_scale(double *data, size_t rows, size_t cols,
                                     const double *means, const double *stddevs);
int brlmm_compute_xtx(const double *data, size_t rows, size_t cols, double *out);
void brlmm_matrix_multiply_raw(const double *A, size_t a_rows, size_t a_cols,
                               const double *B, size_t b_cols, double *out);

#endif /* BRLMM_UTILS_H */
