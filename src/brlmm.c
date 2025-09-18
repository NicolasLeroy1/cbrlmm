#include "brlmm.h"
#include "brlmm_utils.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#if __has_include(<lapacke.h>)
#include <lapacke.h>
#define BRLMM_HAVE_LAPACKE 1
#else
#define BRLMM_HAVE_LAPACKE 0
#endif

#if !BRLMM_HAVE_LAPACKE
typedef int lapack_int;
extern void dgesvd_(char *jobu, char *jobvt, const lapack_int *m, const lapack_int *n,
                    double *a, const lapack_int *lda, double *s,
                    double *u, const lapack_int *ldu, double *vt, const lapack_int *ldvt,
                    double *work, const lapack_int *lwork, lapack_int *info);
extern void dsyev_(char *jobz, char *uplo, const lapack_int *n,
                   double *a, const lapack_int *lda, double *w,
                   double *work, const lapack_int *lwork, lapack_int *info);
#endif

static void brlmm_latent_clear(BrlmmLatent *latent);
static int brlmm_latent_copy(const BrlmmLatent *src, BrlmmLatent *dst);
static void brlmm_latent_array_release(BrlmmLatentArray *array);
static void brlmm_vector_array_free(BrlmmVectorArray *arr);
static int brlmm_vector_array_alloc(BrlmmVectorArray *arr, size_t count);
static int brlmm_vector_array_copy(const BrlmmVectorArray *src, BrlmmVectorArray *dst);
static double brlmm_mean_squared_error(const double *residuals, size_t n);
static double brlmm_safe_variance(const double *values, size_t n);
static void brlmm_input_reset(BrlmmInput *input);
static void brlmm_state_reset(BrlmmState *state);
static void brlmm_sample_reset(BrlmmSample *sample);
static void brlmm_output_reset(BrlmmOutput *output);
static int brlmm_prepare_latents_from_X(const BrlmmMatrix *X,
                                        double inertia_threshold,
                                        bool scale_X,
                                        BrlmmLatent *latent);
static int brlmm_prepare_latents_from_K(const BrlmmMatrix *K,
                                        double inertia_threshold,
                                        BrlmmLatent *latent);
static double brlmm_safe_log(double x);
static int brlmm_allocate_output_storage(const BrlmmInput *input,
                                         size_t sample_count,
                                         BrlmmOutput *out_result);

static void brlmm_latent_clear(BrlmmLatent *latent) {
    if (!latent) {
        return;
    }
    brlmm_free_matrix(&latent->U);
    brlmm_free_matrix(&latent->V);
    brlmm_free_vector(&latent->singular);
    brlmm_free_vector(&latent->eigen);
    brlmm_free_vector(&latent->scaled);
    brlmm_free_vector(&latent->centers);
    brlmm_free_vector(&latent->scale);
    latent->rank = 0;
}

static void brlmm_latent_array_release(BrlmmLatentArray *array) {
    if (!array || !array->items) {
        return;
    }
    for (size_t i = 0; i < array->count; ++i) {
        brlmm_latent_clear(&array->items[i]);
    }
    free(array->items);
    array->items = NULL;
    array->count = 0;
}

static int brlmm_latent_copy(const BrlmmLatent *src, BrlmmLatent *dst) {
    if (!src || !dst) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    dst->kind = src->kind;
    dst->rank = src->rank;
    int rc = BRLMM_OK;
    if (src->U.data) {
        rc = brlmm_allocate_matrix(&dst->U, src->U.rows, src->U.cols);
        if (rc != BRLMM_OK) {
            return rc;
        }
        brlmm_matrix_copy(dst->U.data, src->U.data, src->U.rows * src->U.cols);
    }
    if (src->V.data) {
        rc = brlmm_allocate_matrix(&dst->V, src->V.rows, src->V.cols);
        if (rc != BRLMM_OK) {
            brlmm_latent_clear(dst);
            return rc;
        }
        brlmm_matrix_copy(dst->V.data, src->V.data, src->V.rows * src->V.cols);
    }
    if (src->singular.data) {
        rc = brlmm_allocate_vector(&dst->singular, src->singular.size);
        if (rc != BRLMM_OK) {
            brlmm_latent_clear(dst);
            return rc;
        }
        brlmm_vector_copy(dst->singular.data, src->singular.data, src->singular.size);
    }
    if (src->eigen.data) {
        rc = brlmm_allocate_vector(&dst->eigen, src->eigen.size);
        if (rc != BRLMM_OK) {
            brlmm_latent_clear(dst);
            return rc;
        }
        brlmm_vector_copy(dst->eigen.data, src->eigen.data, src->eigen.size);
    }
    if (src->scaled.data) {
        rc = brlmm_allocate_vector(&dst->scaled, src->scaled.size);
        if (rc != BRLMM_OK) {
            brlmm_latent_clear(dst);
            return rc;
        }
        brlmm_vector_copy(dst->scaled.data, src->scaled.data, src->scaled.size);
    }
    if (src->centers.data) {
        rc = brlmm_allocate_vector(&dst->centers, src->centers.size);
        if (rc != BRLMM_OK) {
            brlmm_latent_clear(dst);
            return rc;
        }
        brlmm_vector_copy(dst->centers.data, src->centers.data, src->centers.size);
    }
    if (src->scale.data) {
        rc = brlmm_allocate_vector(&dst->scale, src->scale.size);
        if (rc != BRLMM_OK) {
            brlmm_latent_clear(dst);
            return rc;
        }
        brlmm_vector_copy(dst->scale.data, src->scale.data, src->scale.size);
    }
    return BRLMM_OK;
}

static void brlmm_vector_array_free(BrlmmVectorArray *arr) {
    if (!arr) {
        return;
    }
    for (size_t i = 0; i < arr->count; ++i) {
        brlmm_free_vector(&arr->items[i]);
    }
    free(arr->items);
    arr->items = NULL;
    arr->count = 0;
}

static int brlmm_vector_array_alloc(BrlmmVectorArray *arr, size_t count) {
    if (!arr) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    arr->items = (BrlmmVector *)calloc(count, sizeof(BrlmmVector));
    if (!arr->items) {
        arr->count = 0;
        return BRLMM_ERR_MEMORY;
    }
    arr->count = count;
    return BRLMM_OK;
}

static int brlmm_vector_array_copy(const BrlmmVectorArray *src, BrlmmVectorArray *dst) {
    if (!src || !dst) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    int rc = brlmm_vector_array_alloc(dst, src->count);
    if (rc != BRLMM_OK) {
        return rc;
    }
    for (size_t i = 0; i < src->count; ++i) {
        if (src->items[i].data && src->items[i].size) {
            rc = brlmm_allocate_vector(&dst->items[i], src->items[i].size);
            if (rc != BRLMM_OK) {
                brlmm_vector_array_free(dst);
                return rc;
            }
            brlmm_vector_copy(dst->items[i].data, src->items[i].data, src->items[i].size);
        }
    }
    return BRLMM_OK;
}

static double brlmm_mean_squared_error(const double *residuals, size_t n) {
    if (!residuals || n == 0) {
        return 0.0;
    }
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += residuals[i] * residuals[i];
    }
    return sum / (double)n;
}

static double brlmm_safe_variance(const double *values, size_t n) {
    double var = brlmm_vector_variance(values, n);
    if (var <= 0.0) {
        return 1.0;
    }
    return var;
}

static void brlmm_input_reset(BrlmmInput *input) {
    if (!input) {
        return;
    }
    input->n = 0;
    input->LX = 0;
    input->LK = 0;
    input->y.data = NULL;
    input->y.size = 0;
    input->latents.count = 0;
    input->latents.items = NULL;
}

static void brlmm_state_reset(BrlmmState *state) {
    if (!state) {
        return;
    }
    state->mu = 0.0;
    state->sigma2 = 0.0;
    state->tau2 = 0.0;
    state->xi2_tau2 = 0.0;
    brlmm_free_vector(&state->residuals);
    brlmm_free_vector(&state->lambda);
    brlmm_free_vector(&state->omega2);
    brlmm_free_vector(&state->xi2_omega2);
    brlmm_vector_array_free(&state->nu);
}

static void brlmm_sample_reset(BrlmmSample *sample) {
    if (!sample) {
        return;
    }
    sample->mu = 0.0;
    sample->sigma2 = 0.0;
    sample->loss = 0.0;
    brlmm_free_vector(&sample->residuals);
    brlmm_free_vector(&sample->lambda);
    brlmm_vector_array_free(&sample->nu);
}

static void brlmm_output_reset(BrlmmOutput *output) {
    if (!output) {
        return;
    }
    output->sample_count = 0;
    output->L = 0;
    output->LX = 0;
    output->LK = 0;
    output->n = 0;
    output->mu_mean = 0.0;
    brlmm_free_vector(&output->mu_chain);
    brlmm_free_vector(&output->sigma2_chain);
    brlmm_free_vector(&output->mse_chain);
    brlmm_free_vector(&output->loss_chain);
    brlmm_free_matrix(&output->lambda_chain);
    brlmm_vector_array_free(&output->nu_chain);
    if (output->latents.items) {
        for (size_t i = 0; i < output->latents.count; ++i) {
            brlmm_latent_clear(&output->latents.items[i]);
        }
        free(output->latents.items);
        output->latents.items = NULL;
        output->latents.count = 0;
    }
}

static double brlmm_safe_log(double x) {
    if (x <= 0.0) {
        return -1e12;
    }
    return log(x);
}

static int brlmm_allocate_output_storage(const BrlmmInput *input,
                                         size_t sample_count,
                                         BrlmmOutput *out_result) {
    if (!input || !out_result || sample_count == 0) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    int rc = brlmm_allocate_vector(&out_result->mu_chain, sample_count);
    if (rc != BRLMM_OK) {
        return rc;
    }
    rc = brlmm_allocate_vector(&out_result->sigma2_chain, sample_count);
    if (rc != BRLMM_OK) {
        return rc;
    }
    rc = brlmm_allocate_vector(&out_result->mse_chain, sample_count);
    if (rc != BRLMM_OK) {
        return rc;
    }
    rc = brlmm_allocate_vector(&out_result->loss_chain, sample_count);
    if (rc != BRLMM_OK) {
        return rc;
    }
    rc = brlmm_allocate_matrix(&out_result->lambda_chain,
                               input->latents.count,
                               sample_count);
    if (rc != BRLMM_OK) {
        return rc;
    }
    rc = brlmm_vector_array_alloc(&out_result->nu_chain, input->latents.count);
    if (rc != BRLMM_OK) {
        return rc;
    }
    for (size_t l = 0; l < input->latents.count; ++l) {
        size_t rank = input->latents.items[l].rank;
        if (rank == 0) {
            continue;
        }
        rc = brlmm_allocate_vector(&out_result->nu_chain.items[l], rank * sample_count);
        if (rc != BRLMM_OK) {
            return rc;
        }
    }
    out_result->nu_chain.count = input->latents.count;
    out_result->sample_count = sample_count;
    out_result->L = input->latents.count;
    out_result->LX = input->LX;
    out_result->LK = input->LK;
    out_result->n = input->n;
    return BRLMM_OK;
}

int brlmm_parse_input(const BrlmmProblem *problem,
                      const BrlmmConfig *config,
                      BrlmmInput *out_input) {
    if (!problem || !config || !out_input) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    brlmm_input_reset(out_input);
    if (config->method != BRLMM_METHOD_HORSESHOE) {
        return BRLMM_ERR_UNSUPPORTED;
    }
    if (!problem->y.data || problem->y.size == 0) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    size_t n = problem->y.size;
    int rc = brlmm_allocate_vector(&out_input->y, n);
    if (rc != BRLMM_OK) {
        return rc;
    }
    brlmm_vector_copy(out_input->y.data, problem->y.data, n);
    out_input->n = n;

    BrlmmLatentArray X_latents = {0};
    BrlmmLatentArray K_latents = {0};

    if (problem->X_list.count == 0 && problem->K_list.count == 0) {
        brlmm_input_clear(out_input);
        return BRLMM_ERR_INVALID_ARGUMENT;
    }

    if (problem->X_list.count > 0) {
        X_latents.items = (BrlmmLatent *)calloc(problem->X_list.count, sizeof(BrlmmLatent));
        if (!X_latents.items) {
            brlmm_latent_array_release(&X_latents);
            brlmm_latent_array_release(&K_latents);
            brlmm_input_clear(out_input);
            return BRLMM_ERR_MEMORY;
        }
        X_latents.count = problem->X_list.count;
        for (size_t i = 0; i < problem->X_list.count; ++i) {
            const BrlmmMatrix *X = &problem->X_list.items[i];
            if (!X->data || X->rows != n) {
                brlmm_latent_array_release(&X_latents);
                brlmm_latent_array_release(&K_latents);
                brlmm_input_clear(out_input);
                return BRLMM_ERR_DIMENSION_MISMATCH;
            }
            rc = brlmm_prepare_latents_from_X(X,
                                              config->inertia_threshold,
                                              config->scale_X,
                                              &X_latents.items[i]);
            if (rc != BRLMM_OK) {
                brlmm_latent_array_release(&X_latents);
                brlmm_latent_array_release(&K_latents);
                brlmm_input_clear(out_input);
                return rc;
            }
        }
    }

    if (problem->K_list.count > 0) {
        K_latents.items = (BrlmmLatent *)calloc(problem->K_list.count, sizeof(BrlmmLatent));
        if (!K_latents.items) {
            brlmm_latent_array_release(&X_latents);
            brlmm_latent_array_release(&K_latents);
            brlmm_input_clear(out_input);
            return BRLMM_ERR_MEMORY;
        }
        K_latents.count = problem->K_list.count;
        for (size_t i = 0; i < problem->K_list.count; ++i) {
            const BrlmmMatrix *K = &problem->K_list.items[i];
            if (!K->data || K->rows != n || K->cols != n) {
                brlmm_latent_array_release(&X_latents);
                brlmm_latent_array_release(&K_latents);
                brlmm_input_clear(out_input);
                return BRLMM_ERR_DIMENSION_MISMATCH;
            }
            rc = brlmm_prepare_latents_from_K(K,
                                              config->inertia_threshold,
                                              &K_latents.items[i]);
            if (rc != BRLMM_OK) {
                brlmm_latent_array_release(&X_latents);
                brlmm_latent_array_release(&K_latents);
                brlmm_input_clear(out_input);
                return rc;
            }
        }
    }

    size_t L = X_latents.count + K_latents.count;
    out_input->latents.items = (BrlmmLatent *)calloc(L, sizeof(BrlmmLatent));
    if (!out_input->latents.items) {
        brlmm_latent_array_release(&X_latents);
        brlmm_latent_array_release(&K_latents);
        brlmm_input_clear(out_input);
        return BRLMM_ERR_MEMORY;
    }
    out_input->latents.count = L;
    out_input->LX = X_latents.count;
    out_input->LK = K_latents.count;

    for (size_t i = 0; i < X_latents.count; ++i) {
        rc = brlmm_latent_copy(&X_latents.items[i], &out_input->latents.items[i]);
        if (rc != BRLMM_OK) {
            brlmm_latent_array_release(&X_latents);
            brlmm_latent_array_release(&K_latents);
            brlmm_input_clear(out_input);
            return rc;
        }
    }
    for (size_t i = 0; i < K_latents.count; ++i) {
        rc = brlmm_latent_copy(&K_latents.items[i], &out_input->latents.items[out_input->LX + i]);
        if (rc != BRLMM_OK) {
            brlmm_latent_array_release(&X_latents);
            brlmm_latent_array_release(&K_latents);
            brlmm_input_clear(out_input);
            return rc;
        }
    }

    brlmm_latent_array_release(&X_latents);
    brlmm_latent_array_release(&K_latents);

    return BRLMM_OK;
}

void brlmm_input_clear(BrlmmInput *input) {
    if (!input) {
        return;
    }
    brlmm_free_vector(&input->y);
    if (input->latents.items) {
        for (size_t i = 0; i < input->latents.count; ++i) {
            brlmm_latent_clear(&input->latents.items[i]);
        }
        free(input->latents.items);
    }
    input->latents.items = NULL;
    input->latents.count = 0;
    input->LX = 0;
    input->LK = 0;
    input->n = 0;
}

int brlmm_state_init(const BrlmmInput *input,
                     BrlmmMethod method,
                     BrlmmState *out_state) {
    if (!input || !out_state || method != BRLMM_METHOD_HORSESHOE) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    brlmm_state_reset(out_state);
    size_t n = input->n;
    size_t L = input->latents.count;
    out_state->mu = brlmm_vector_mean(input->y.data, n);
    double variance = brlmm_safe_variance(input->y.data, n);
    out_state->sigma2 = variance;
    out_state->tau2 = 1.0;
    out_state->xi2_tau2 = 1.0;

    int rc = brlmm_allocate_vector(&out_state->residuals, n);
    if (rc != BRLMM_OK) {
        return rc;
    }
    for (size_t i = 0; i < n; ++i) {
        out_state->residuals.data[i] = input->y.data[i] - out_state->mu;
    }

    rc = brlmm_allocate_vector(&out_state->lambda, L);
    if (rc != BRLMM_OK) {
        brlmm_state_clear(out_state);
        return rc;
    }
    for (size_t i = 0; i < L; ++i) {
        out_state->lambda.data[i] = 1.0;
    }

    rc = brlmm_allocate_vector(&out_state->omega2, L);
    if (rc != BRLMM_OK) {
        brlmm_state_clear(out_state);
        return rc;
    }
    rc = brlmm_allocate_vector(&out_state->xi2_omega2, L);
    if (rc != BRLMM_OK) {
        brlmm_state_clear(out_state);
        return rc;
    }
    for (size_t i = 0; i < L; ++i) {
        out_state->omega2.data[i] = 1.0;
        out_state->xi2_omega2.data[i] = 1.0;
    }

    rc = brlmm_vector_array_alloc(&out_state->nu, L);
    if (rc != BRLMM_OK) {
        brlmm_state_clear(out_state);
        return rc;
    }
    for (size_t l = 0; l < L; ++l) {
        size_t rank = input->latents.items[l].rank;
        if (rank == 0) {
            continue;
        }
        rc = brlmm_allocate_vector(&out_state->nu.items[l], rank);
        if (rc != BRLMM_OK) {
            brlmm_state_clear(out_state);
            return rc;
        }
        brlmm_vector_zero(out_state->nu.items[l].data, rank);
    }

    return BRLMM_OK;
}

void brlmm_state_clear(BrlmmState *state) {
    brlmm_state_reset(state);
}

static void brlmm_update_residual_with_effect(double *residuals,
                                              const double *effect,
                                              double scale,
                                              size_t n) {
    if (!residuals || !effect) {
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        residuals[i] += scale * effect[i];
    }
}

int brlmm_update_state_horseshoe(BrlmmState *state,
                                 const BrlmmLatentArray *latents,
                                 const BrlmmRng *rng) {
    if (!state || !latents || !rng) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    size_t n = state->residuals.size;
    size_t L = latents->count;
    if (L == 0) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }

    double sigma2 = state->sigma2;
    double tau2 = state->tau2;
    double mu_old = state->mu;

    double *residuals = state->residuals.data;
    double sum_y = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double value = residuals[i] + mu_old;
        residuals[i] = value;
        sum_y += value;
    }
    double mean_y = sum_y / (double)n;
    double mu_new = brlmm_rng_normal(rng, mean_y, sqrt(sigma2 / (double)n));
    state->mu = mu_new;
    for (size_t i = 0; i < n; ++i) {
        residuals[i] -= mu_new;
    }

    double *lambda_vec = state->lambda.data;
    double *omega2_vec = state->omega2.data;
    double *xi2_omega2_vec = state->xi2_omega2.data;

    double *effect_buffer = (double *)calloc(n, sizeof(double));
    if (!effect_buffer) {
        return BRLMM_ERR_MEMORY;
    }
    size_t *order = (size_t *)malloc(L * sizeof(size_t));
    if (!order) {
        free(effect_buffer);
        return BRLMM_ERR_MEMORY;
    }
    int shuffle_rc = brlmm_shuffle_indices(L, order, rng);
    if (shuffle_rc != BRLMM_OK) {
        free(order);
        free(effect_buffer);
        return shuffle_rc;
    }

    size_t max_rank = 0;
    for (size_t l = 0; l < L; ++l) {
        if (state->nu.items[l].size > max_rank) {
            max_rank = state->nu.items[l].size;
        }
    }

    double *UtR = NULL;
    double *scale2_nu = NULL;
    double *center_nu = NULL;
    if (max_rank > 0) {
        UtR = (double *)calloc(max_rank, sizeof(double));
        scale2_nu = (double *)calloc(max_rank, sizeof(double));
        center_nu = (double *)calloc(max_rank, sizeof(double));
        if (!UtR || !scale2_nu || !center_nu) {
            free(center_nu);
            free(scale2_nu);
            free(UtR);
            free(order);
            free(effect_buffer);
            return BRLMM_ERR_MEMORY;
        }
    }

    int loop_status = BRLMM_OK;
    for (size_t idx = 0; idx < L; ++idx) {
        size_t l = order[idx];
        const BrlmmLatent *latent = &latents->items[l];
        double *nu_l = state->nu.items[l].data;
        size_t rank = state->nu.items[l].size;
        if (!latent->U.data || rank == 0) {
            continue;
        }

        double lambda_old = lambda_vec[l];
        brlmm_matrix_vector_mul(&latent->U, nu_l, effect_buffer);
        brlmm_update_residual_with_effect(residuals, effect_buffer, lambda_old, n);

        if (UtR) {
            memset(UtR, 0, max_rank * sizeof(double));
            memset(scale2_nu, 0, max_rank * sizeof(double));
            memset(center_nu, 0, max_rank * sizeof(double));
        }

        brlmm_matrix_transpose_vector_mul(&latent->U, residuals, UtR);
        const double *sD2 = latent->scaled.data;
        for (size_t j = 0; j < rank; ++j) {
            double s_val = (sD2 && sD2[j] > 0.0) ? sD2[j] : 1e-6;
            double denom = (1.0 / s_val) + (lambda_old * lambda_old) / sigma2;
            scale2_nu[j] = 1.0 / denom;
            center_nu[j] = lambda_old * scale2_nu[j] * UtR[j] / sigma2;
            double sd = sqrt(fmax(scale2_nu[j], 0.0));
            nu_l[j] = brlmm_rng_normal(rng, center_nu[j], sd);
        }

        double sum_nu_sq = 0.0;
        double sum_UtRnu = 0.0;
        for (size_t j = 0; j < rank; ++j) {
            sum_nu_sq += nu_l[j] * nu_l[j];
            sum_UtRnu += UtR[j] * nu_l[j];
        }
        double omega2 = omega2_vec[l];
        double scale2_lambda = 1.0 /
            (sum_nu_sq / sigma2 + 1.0 / (sigma2 * tau2 * omega2));
        double sd_lambda = sqrt(fmax(scale2_lambda, 0.0));
        double center_lambda = scale2_lambda * sum_UtRnu / sigma2;
        double lambda_new = brlmm_rng_normal(rng, center_lambda, sd_lambda);

        double b_omega2 = lambda_new * lambda_new /
            (2.0 * sigma2 * tau2) + 1.0 / xi2_omega2_vec[l];
        omega2 = brlmm_rng_invgamma(rng, 1.0, b_omega2);
        if (omega2 <= 0.0) {
            omega2 = 1e-6;
        }
        double b_xi = 1.0 + 1.0 / omega2;
        double xi2_omega2 = brlmm_rng_invgamma(rng, 1.0, b_xi);
        if (xi2_omega2 <= 0.0) {
            xi2_omega2 = 1e-6;
        }

        lambda_vec[l] = lambda_new;
        omega2_vec[l] = omega2;
        xi2_omega2_vec[l] = xi2_omega2;

        brlmm_matrix_vector_mul(&latent->U, nu_l, effect_buffer);
        brlmm_update_residual_with_effect(residuals, effect_buffer, -lambda_new, n);
    }

    free(order);
    free(effect_buffer);
    free(center_nu);
    free(scale2_nu);
    free(UtR);
    if (loop_status != BRLMM_OK) {
        return loop_status;
    }

    double sum_lambda_ratio = 0.0;
    for (size_t l = 0; l < L; ++l) {
        sum_lambda_ratio += (lambda_vec[l] * lambda_vec[l]) /
            omega2_vec[l];
    }
    double a_tau2 = 0.5 * ((double)L + 1.0);
    double b_tau2 = sum_lambda_ratio / (2.0 * sigma2) + 1.0 / state->xi2_tau2;
    tau2 = brlmm_rng_invgamma(rng, a_tau2, b_tau2);
    if (tau2 <= 0.0) {
        tau2 = 1e-6;
    }
    state->tau2 = tau2;

    double b_xi_tau2 = 1.0 + 1.0 / tau2;
    double xi2_tau2 = brlmm_rng_invgamma(rng, 1.0, b_xi_tau2);
    if (xi2_tau2 <= 0.0) {
        xi2_tau2 = 1e-6;
    }
    state->xi2_tau2 = xi2_tau2;

    double sum_residual_sq = brlmm_vector_sum_sq(residuals, n);
    double sum_lambda_over = 0.0;
    for (size_t l = 0; l < L; ++l) {
        sum_lambda_over += (lambda_vec[l] * lambda_vec[l]) /
            omega2_vec[l];
    }
    double a_sig2 = 0.5 * ((double)n + (double)L);
    double b_sig2 = sum_residual_sq / 2.0 + sum_lambda_over / (2.0 * tau2);
    sigma2 = brlmm_rng_invgamma(rng, a_sig2, b_sig2);
    if (sigma2 <= 0.0) {
        sigma2 = 1e-6;
    }
    state->sigma2 = sigma2;

    return BRLMM_OK;
}

int brlmm_compute_loss(const BrlmmInput *input,
                       const BrlmmState *state,
                       BrlmmMethod method,
                       double *out_loss) {
    if (!input || !state || !out_loss || method != BRLMM_METHOD_HORSESHOE) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    size_t L = input->latents.count;
    size_t n = input->n;
    const double *residuals = state->residuals.data;
    const double *lambda = state->lambda.data;
    const double *omega2 = state->omega2.data;
    const double *xi2_omega2 = state->xi2_omega2.data;
    double sigma2 = state->sigma2;
    double tau2 = state->tau2;
    double xi2_tau2 = state->xi2_tau2;
    if (sigma2 <= 0.0 || tau2 <= 0.0 || xi2_tau2 <= 0.0) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }

    double residual_term = brlmm_vector_sum_sq(residuals, n) / (2.0 * sigma2);

    double latent_penalty = 0.0;
    for (size_t l = 0; l < L; ++l) {
        const BrlmmVector *nu = &state->nu.items[l];
        const BrlmmLatent *latent = &input->latents.items[l];
        for (size_t j = 0; j < nu->size; ++j) {
            double sD2 = (latent->scaled.data && latent->scaled.data[j] > 0.0)
                ? latent->scaled.data[j]
                : 1e-6;
            latent_penalty += (nu->data[j] * nu->data[j]) / (2.0 * sD2);
        }
    }

    double lambda_penalty = 0.0;
    double omega_penalty = 0.0;
    double omega_xi_terms = 0.0;
    for (size_t l = 0; l < L; ++l) {
        if (omega2[l] <= 0.0 || xi2_omega2[l] <= 0.0) {
            return BRLMM_ERR_INVALID_ARGUMENT;
        }
        lambda_penalty += (lambda[l] * lambda[l]) /
            (2.0 * tau2 * sigma2 * omega2[l]);
        omega_penalty += 2.0 * brlmm_safe_log(omega2[l]);
        omega_xi_terms += 1.0 / (omega2[l] * xi2_omega2[l]);
        omega_xi_terms += 1.0 / xi2_omega2[l];
        omega_xi_terms += 2.0 * brlmm_safe_log(xi2_omega2[l]);
    }

    double tau_log_term = ((double)L + 3.0) * brlmm_safe_log(tau2) / 2.0;
    double sigma_log_term = ((double)n + (double)L + 2.0) * brlmm_safe_log(sigma2) / 2.0;
    double tau_xi_terms = 1.0 / (tau2 * xi2_tau2) + 1.0 / xi2_tau2 +
        2.0 * brlmm_safe_log(xi2_tau2);

    *out_loss = residual_term + latent_penalty + lambda_penalty + omega_penalty +
        tau_log_term + sigma_log_term + tau_xi_terms + omega_xi_terms;
    return BRLMM_OK;
}

int brlmm_record_sample(const BrlmmInput *input,
                        const BrlmmState *state,
                        BrlmmSample *out_sample) {
    if (!input || !state || !out_sample) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    brlmm_sample_reset(out_sample);
    out_sample->mu = state->mu;
    out_sample->sigma2 = state->sigma2;

    int rc = brlmm_allocate_vector(&out_sample->residuals, input->n);
    if (rc != BRLMM_OK) {
        return rc;
    }
    brlmm_vector_copy(out_sample->residuals.data,
                      state->residuals.data,
                      input->n);

    rc = brlmm_allocate_vector(&out_sample->lambda, input->latents.count);
    if (rc != BRLMM_OK) {
        brlmm_sample_clear(out_sample);
        return rc;
    }
    brlmm_vector_copy(out_sample->lambda.data,
                      state->lambda.data,
                      input->latents.count);

    rc = brlmm_vector_array_alloc(&out_sample->nu, input->latents.count);
    if (rc != BRLMM_OK) {
        brlmm_sample_clear(out_sample);
        return rc;
    }
    for (size_t l = 0; l < input->latents.count; ++l) {
        size_t rank = state->nu.items[l].size;
        if (rank == 0) {
            continue;
        }
        rc = brlmm_allocate_vector(&out_sample->nu.items[l], rank);
        if (rc != BRLMM_OK) {
            brlmm_sample_clear(out_sample);
            return rc;
        }
        brlmm_vector_copy(out_sample->nu.items[l].data,
                          state->nu.items[l].data,
                          rank);
    }

    rc = brlmm_compute_loss(input, state, BRLMM_METHOD_HORSESHOE, &out_sample->loss);
    if (rc != BRLMM_OK) {
        brlmm_sample_clear(out_sample);
        return rc;
    }

    return BRLMM_OK;
}

void brlmm_sample_clear(BrlmmSample *sample) {
    brlmm_sample_reset(sample);
}

int brlmm_run(const BrlmmProblem *problem,
              const BrlmmConfig *config,
              const BrlmmRng *rng,
              BrlmmOutput *out_result) {
    if (!problem || !config || !rng || !out_result) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    brlmm_output_reset(out_result);

    BrlmmInput input;
    brlmm_input_reset(&input);
    int rc = brlmm_parse_input(problem, config, &input);
    if (rc != BRLMM_OK) {
        return rc;
    }

    BrlmmState state;
    memset(&state, 0, sizeof(state));
    rc = brlmm_state_init(&input, config->method, &state);
    if (rc != BRLMM_OK) {
        brlmm_input_clear(&input);
        return rc;
    }

    size_t n_iter = config->n_iter;
    size_t burnin = config->burnin;
    size_t thinning = config->thinning;
    size_t sample_count = (n_iter - burnin) / thinning;
    if (sample_count == 0) {
        brlmm_state_clear(&state);
        brlmm_input_clear(&input);
        return BRLMM_ERR_INVALID_ARGUMENT;
    }

    rc = brlmm_allocate_output_storage(&input, sample_count, out_result);
    if (rc != BRLMM_OK) {
        brlmm_state_clear(&state);
        brlmm_input_clear(&input);
        return rc;
    }

    out_result->latents.items = (BrlmmLatent *)calloc(input.latents.count, sizeof(BrlmmLatent));
    if (!out_result->latents.items) {
        brlmm_state_clear(&state);
        brlmm_input_clear(&input);
        return BRLMM_ERR_MEMORY;
    }
    out_result->latents.count = input.latents.count;
    for (size_t l = 0; l < input.latents.count; ++l) {
        rc = brlmm_latent_copy(&input.latents.items[l], &out_result->latents.items[l]);
        if (rc != BRLMM_OK) {
            brlmm_state_clear(&state);
            brlmm_input_clear(&input);
            return rc;
        }
    }

    size_t stored = 0;
    for (size_t iter = 0; iter < n_iter; ++iter) {
        rc = brlmm_update_state_horseshoe(&state, &input.latents, rng);
        if (rc != BRLMM_OK) {
            brlmm_state_clear(&state);
            brlmm_input_clear(&input);
            brlmm_output_clear(out_result);
            return rc;
        }
        if (iter < burnin) {
            continue;
        }
        size_t post_burn = iter - burnin;
        if (post_burn % thinning != 0) {
            continue;
        }
        if (stored >= sample_count) {
            break;
        }
        out_result->mu_chain.data[stored] = state.mu;
        out_result->sigma2_chain.data[stored] = state.sigma2;
        out_result->mse_chain.data[stored] = brlmm_mean_squared_error(state.residuals.data, input.n);
        rc = brlmm_compute_loss(&input, &state, BRLMM_METHOD_HORSESHOE, &out_result->loss_chain.data[stored]);
        if (rc != BRLMM_OK) {
            brlmm_state_clear(&state);
            brlmm_input_clear(&input);
            brlmm_output_clear(out_result);
            return rc;
        }
        for (size_t l = 0; l < input.latents.count; ++l) {
            out_result->lambda_chain.data[l * sample_count + stored] = state.lambda.data[l];
            size_t rank = state.nu.items[l].size;
            if (rank == 0) {
                continue;
            }
            for (size_t j = 0; j < rank; ++j) {
                out_result->nu_chain.items[l].data[j * sample_count + stored] =
                    state.nu.items[l].data[j];
            }
        }
        ++stored;
    }

    double sum_mu = 0.0;
    for (size_t i = 0; i < stored; ++i) {
        sum_mu += out_result->mu_chain.data[i];
    }
    if (stored > 0) {
        out_result->mu_mean = sum_mu / (double)stored;
    } else {
        out_result->mu_mean = 0.0;
    }

    out_result->sample_count = stored;
    out_result->mu_chain.size = stored;
    out_result->sigma2_chain.size = stored;
    out_result->mse_chain.size = stored;
    out_result->loss_chain.size = stored;
    out_result->lambda_chain.cols = stored;
    for (size_t l = 0; l < out_result->nu_chain.count; ++l) {
        size_t rank = input.latents.items[l].rank;
        out_result->nu_chain.items[l].size = rank * stored;
    }

    brlmm_state_clear(&state);
    brlmm_input_clear(&input);

    return BRLMM_OK;
}

void brlmm_output_clear(BrlmmOutput *output) {
    brlmm_output_reset(output);
}

static int brlmm_predict_fixed(const BrlmmLatent *latent,
                               const BrlmmMatrix *new_X,
                               BrlmmMatrix *out_U) {
    size_t n = new_X->rows;
    size_t p = new_X->cols;
    size_t rank = latent->rank;
    if (latent->V.cols != rank || latent->V.rows != p) {
        return BRLMM_ERR_DIMENSION_MISMATCH;
    }
    int rc = brlmm_allocate_matrix(out_U, n, rank);
    if (rc != BRLMM_OK) {
        return rc;
    }

    double *work = (double *)malloc(n * p * sizeof(double));
    if (!work) {
        brlmm_free_matrix(out_U);
        return BRLMM_ERR_MEMORY;
    }
    brlmm_matrix_copy(work, new_X->data, n * p);
    brlmm_apply_column_center_scale(work, n, p,
                                    latent->centers.data,
                                    latent->scale.data);
    double *XV = (double *)calloc(n * rank, sizeof(double));
    if (!XV) {
        free(work);
        brlmm_free_matrix(out_U);
        return BRLMM_ERR_MEMORY;
    }
    brlmm_matrix_multiply_raw(work, n, p,
                              latent->V.data, rank,
                              XV);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < rank; ++j) {
            double d = latent->singular.data[j];
            double value = (d != 0.0) ? XV[i * rank + j] / d : 0.0;
            out_U->data[i * rank + j] = value;
        }
    }
    free(work);
    free(XV);
    return BRLMM_OK;
}

static int brlmm_predict_random(const BrlmmLatent *latent,
                                const BrlmmMatrix *new_K,
                                size_t original_n,
                                BrlmmMatrix *out_U) {
    size_t n_new = new_K->rows;
    size_t n_cols = new_K->cols;
    size_t rank = latent->rank;
    if (n_cols != original_n) {
        return BRLMM_ERR_DIMENSION_MISMATCH;
    }
    int rc = brlmm_allocate_matrix(out_U, n_new, rank);
    if (rc != BRLMM_OK) {
        return rc;
    }
    brlmm_matrix_multiply_raw(new_K->data, n_new, n_cols,
                              latent->U.data, rank,
                              out_U->data);
    for (size_t i = 0; i < n_new; ++i) {
        for (size_t j = 0; j < rank; ++j) {
            double d2 = latent->eigen.data[j];
            double value = (d2 != 0.0) ? out_U->data[i * rank + j] / d2 : 0.0;
            out_U->data[i * rank + j] = value;
        }
    }
    return BRLMM_OK;
}

int brlmm_predict(const BrlmmOutput *fit,
                  const BrlmmMatrixList *new_X_list,
                  const BrlmmMatrixList *new_K_list,
                  BrlmmMatrix *out_chain) {
    if (!fit || !out_chain) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    size_t LX = fit->LX;
    size_t LK = fit->LK;
    size_t L = fit->L;
    size_t sample_count = fit->sample_count;
    if (sample_count == 0) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    bool has_X = new_X_list && new_X_list->count > 0;
    bool has_K = new_K_list && new_K_list->count > 0;
    if (!has_X && !has_K) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    if (has_X && new_X_list->count != LX) {
        return BRLMM_ERR_DIMENSION_MISMATCH;
    }
    if (has_K && new_K_list->count != LK) {
        return BRLMM_ERR_DIMENSION_MISMATCH;
    }

    size_t n_new = 0;
    if (has_X) {
        n_new = new_X_list->items[0].rows;
    } else if (has_K) {
        n_new = new_K_list->items[0].rows;
    }
    if (!out_chain->data || out_chain->rows != n_new || out_chain->cols != sample_count) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }

    BrlmmMatrix *new_U_list = NULL;
    if (L > 0) {
        new_U_list = (BrlmmMatrix *)calloc(L, sizeof(BrlmmMatrix));
        if (!new_U_list) {
            return BRLMM_ERR_MEMORY;
        }
    }

    for (size_t l = 0; l < LX; ++l) {
        int rc = brlmm_predict_fixed(&fit->latents.items[l],
                                     &new_X_list->items[l],
                                     &new_U_list[l]);
        if (rc != BRLMM_OK) {
            for (size_t i = 0; i < L; ++i) {
                brlmm_free_matrix(&new_U_list[i]);
            }
            free(new_U_list);
            return rc;
        }
    }

    for (size_t l = 0; l < LK; ++l) {
        int rc = brlmm_predict_random(&fit->latents.items[LX + l],
                                      &new_K_list->items[l],
                                      fit->n,
                                      &new_U_list[LX + l]);
        if (rc != BRLMM_OK) {
            for (size_t i = 0; i < L; ++i) {
                brlmm_free_matrix(&new_U_list[i]);
            }
            free(new_U_list);
            return rc;
        }
    }

    size_t max_rank = 0;
    for (size_t l = 0; l < L; ++l) {
        if (fit->latents.items[l].rank > max_rank) {
            max_rank = fit->latents.items[l].rank;
        }
    }
    double *effect = (double *)calloc(max_rank, sizeof(double));
    double *tmp = (double *)calloc(n_new, sizeof(double));
    if ((!effect && max_rank > 0) || !tmp) {
        free(effect);
        free(tmp);
        for (size_t i = 0; i < L; ++i) {
            brlmm_free_matrix(&new_U_list[i]);
        }
        free(new_U_list);
        return BRLMM_ERR_MEMORY;
    }

    for (size_t s = 0; s < sample_count; ++s) {
        for (size_t i = 0; i < n_new; ++i) {
            out_chain->data[i * sample_count + s] = fit->mu_chain.data[s];
        }
        if (has_X) {
            for (size_t l = 0; l < LX; ++l) {
                size_t rank = fit->latents.items[l].rank;
                double lambda = fit->lambda_chain.data[l * sample_count + s];
                for (size_t j = 0; j < rank; ++j) {
                    effect[j] = fit->nu_chain.items[l].data[j * sample_count + s] * lambda;
                }
                brlmm_matrix_vector_mul(&new_U_list[l], effect, tmp);
                for (size_t i = 0; i < n_new; ++i) {
                    out_chain->data[i * sample_count + s] += tmp[i];
                }
            }
        }
        if (has_K) {
            for (size_t l = 0; l < LK; ++l) {
                size_t idx = LX + l;
                size_t rank = fit->latents.items[idx].rank;
                double lambda = fit->lambda_chain.data[idx * sample_count + s];
                for (size_t j = 0; j < rank; ++j) {
                    effect[j] = fit->nu_chain.items[idx].data[j * sample_count + s] * lambda;
                }
                brlmm_matrix_vector_mul(&new_U_list[idx], effect, tmp);
                for (size_t i = 0; i < n_new; ++i) {
                    out_chain->data[i * sample_count + s] += tmp[i];
                }
            }
        }
    }

    free(effect);
    free(tmp);
    for (size_t i = 0; i < L; ++i) {
        brlmm_free_matrix(&new_U_list[i]);
    }
    free(new_U_list);
    return BRLMM_OK;
}

int brlmm_compute_latents_from_X(const BrlmmMatrixList *X_list,
                                 double inertia_threshold,
                                 bool scale_X,
                                 size_t n,
                                 BrlmmLatentArray *out_latents) {
    if (!X_list || !out_latents) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    out_latents->count = X_list->count;
    if (out_latents->count == 0) {
        out_latents->items = NULL;
        return BRLMM_OK;
    }
    out_latents->items = (BrlmmLatent *)calloc(out_latents->count, sizeof(BrlmmLatent));
    if (!out_latents->items) {
        return BRLMM_ERR_MEMORY;
    }
    for (size_t i = 0; i < out_latents->count; ++i) {
        if (X_list->items[i].rows != n) {
            brlmm_latent_array_release(out_latents);
            return BRLMM_ERR_DIMENSION_MISMATCH;
        }
        int rc = brlmm_prepare_latents_from_X(&X_list->items[i], inertia_threshold,
                                              scale_X, &out_latents->items[i]);
        if (rc != BRLMM_OK) {
            brlmm_latent_array_release(out_latents);
            return rc;
        }
    }
    return BRLMM_OK;
}

int brlmm_compute_latents_from_K(const BrlmmMatrixList *K_list,
                                 double inertia_threshold,
                                 size_t n,
                                 BrlmmLatentArray *out_latents) {
    if (!K_list || !out_latents) {
        return BRLMM_ERR_INVALID_ARGUMENT;
    }
    out_latents->count = K_list->count;
    if (out_latents->count == 0) {
        out_latents->items = NULL;
        return BRLMM_OK;
    }
    out_latents->items = (BrlmmLatent *)calloc(out_latents->count, sizeof(BrlmmLatent));
    if (!out_latents->items) {
        return BRLMM_ERR_MEMORY;
    }
    for (size_t i = 0; i < out_latents->count; ++i) {
        if (K_list->items[i].rows != n || K_list->items[i].cols != n) {
            brlmm_latent_array_release(out_latents);
            return BRLMM_ERR_DIMENSION_MISMATCH;
        }
        int rc = brlmm_prepare_latents_from_K(&K_list->items[i], inertia_threshold,
                                              &out_latents->items[i]);
        if (rc != BRLMM_OK) {
            brlmm_latent_array_release(out_latents);
            return rc;
        }
    }
    return BRLMM_OK;
}

static int brlmm_prepare_latents_from_X(const BrlmmMatrix *X,
                                        double inertia_threshold,
                                        bool scale_X,
                                        BrlmmLatent *latent) {
    size_t n = X->rows;
    size_t p = X->cols;
    double *work = (double *)malloc(n * p * sizeof(double));
    if (!work) {
        return BRLMM_ERR_MEMORY;
    }
    brlmm_matrix_copy(work, X->data, n * p);

    double *means = (double *)calloc(p, sizeof(double));
    double *sds = (double *)calloc(p, sizeof(double));
    if (!means || !sds) {
        free(work);
        free(means);
        free(sds);
        return BRLMM_ERR_MEMORY;
    }
    if (scale_X) {
        BrlmmMatrix temp = {n, p, work};
        brlmm_compute_column_stats(&temp, means, sds);
        brlmm_apply_column_center_scale(work, n, p, means, sds);
    }

    double *colX = NULL;
    if (brlmm_alloc_col_major(work, n, p, &colX) != BRLMM_OK) {
        free(work);
        free(means);
        free(sds);
        return BRLMM_ERR_MEMORY;
    }

    lapack_int m = (lapack_int)n;
    lapack_int ncols = (lapack_int)p;
    lapack_int lda = m;
    lapack_int minmn = (m < ncols) ? m : ncols;
    double *singular = (double *)calloc((size_t)minmn, sizeof(double));
    double *u = (double *)calloc((size_t)m * (size_t)minmn, sizeof(double));
    double *vt = (double *)calloc((size_t)minmn * (size_t)ncols, sizeof(double));
    if (!singular || !u || !vt) {
        free(work);
        free(means);
        free(sds);
        free(colX);
        free(singular);
        free(u);
        free(vt);
        return BRLMM_ERR_MEMORY;
    }

    lapack_int info = 0;
#if BRLMM_HAVE_LAPACKE
    info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S',
                          m, ncols, colX, lda,
                          singular, u, m, vt, minmn, NULL);
#else
    lapack_int lwork = -1;
    double wkopt;
    dgesvd_("S", "S", &m, &ncols, colX, &lda, singular,
            u, &m, vt, &minmn, &wkopt, &lwork, &info);
    if (info == 0) {
        lwork = (lapack_int)wkopt;
        double *work_svd = (double *)malloc((size_t)lwork * sizeof(double));
        if (!work_svd) {
            free(work);
            free(means);
            free(sds);
            free(colX);
            free(singular);
            free(u);
            free(vt);
            return BRLMM_ERR_MEMORY;
        }
        dgesvd_("S", "S", &m, &ncols, colX, &lda, singular,
                u, &m, vt, &minmn, work_svd, &lwork, &info);
        free(work_svd);
    }
#endif
    free(colX);
    if (info != 0) {
        free(work);
        free(means);
        free(sds);
        free(singular);
        free(u);
        free(vt);
        return BRLMM_ERR_UNSUPPORTED;
    }

    double total = 0.0;
    for (int j = 0; j < minmn; ++j) {
        double s = singular[j];
        total += s * s;
    }
    if (total <= 0.0) {
        total = 1.0;
    }
    double cumulative = 0.0;
    size_t rank = 0;
    double threshold = fmax(fmin(inertia_threshold, 0.999999), 0.0);
    for (int j = 0; j < minmn; ++j) {
        cumulative += (singular[j] * singular[j]) / total;
        if (cumulative >= threshold) {
            rank = (size_t)j + 1;
            break;
        }
    }
    if (rank == 0) {
        rank = (minmn > 0) ? 1 : 0;
    }

    int rc = brlmm_allocate_matrix(&latent->U, n, rank);
    if (rc != BRLMM_OK) {
        free(work);
        free(means);
        free(sds);
        free(singular);
        free(u);
        free(vt);
        return rc;
    }
    rc = brlmm_allocate_matrix(&latent->V, p, rank);
    if (rc != BRLMM_OK) {
        brlmm_latent_clear(latent);
        free(work);
        free(means);
        free(sds);
        free(singular);
        free(u);
        free(vt);
        return rc;
    }
    rc = brlmm_allocate_vector(&latent->singular, rank);
    if (rc != BRLMM_OK) {
        brlmm_latent_clear(latent);
        free(work);
        free(means);
        free(sds);
        free(singular);
        free(u);
        free(vt);
        return rc;
    }
    rc = brlmm_allocate_vector(&latent->scaled, rank);
    if (rc != BRLMM_OK) {
        brlmm_latent_clear(latent);
        free(work);
        free(means);
        free(sds);
        free(singular);
        free(u);
        free(vt);
        return rc;
    }
    rc = brlmm_allocate_vector(&latent->centers, p);
    if (rc != BRLMM_OK) {
        brlmm_latent_clear(latent);
        free(work);
        free(means);
        free(sds);
        free(singular);
        free(u);
        free(vt);
        return rc;
    }
    rc = brlmm_allocate_vector(&latent->scale, p);
    if (rc != BRLMM_OK) {
        brlmm_latent_clear(latent);
        free(work);
        free(means);
        free(sds);
        free(singular);
        free(u);
        free(vt);
        return rc;
    }
    rc = brlmm_allocate_vector(&latent->eigen, rank);
    if (rc != BRLMM_OK) {
        brlmm_latent_clear(latent);
        free(work);
        free(means);
        free(sds);
        free(singular);
        free(u);
        free(vt);
        return rc;
    }

    latent->kind = BRLMM_LATENT_FIXED_EFFECT;
    latent->rank = rank;

    if (scale_X) {
        brlmm_vector_copy(latent->centers.data, means, p);
        brlmm_vector_copy(latent->scale.data, sds, p);
    } else {
        brlmm_vector_zero(latent->centers.data, p);
        for (size_t j = 0; j < p; ++j) {
            latent->scale.data[j] = 1.0;
        }
    }

    double sum_selected = 0.0;
    for (size_t j = 0; j < rank; ++j) {
        double val = singular[j] * singular[j];
        latent->singular.data[j] = singular[j];
        latent->eigen.data[j] = val;
        sum_selected += val;
    }
    if (sum_selected <= 0.0) {
        sum_selected = 1.0;
    }
    for (size_t j = 0; j < rank; ++j) {
        latent->scaled.data[j] = ((double)n * latent->eigen.data[j]) / sum_selected;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < rank; ++j) {
            latent->U.data[i * rank + j] = u[i + j * m];
        }
    }
    for (size_t r = 0; r < p; ++r) {
        for (size_t j = 0; j < rank; ++j) {
            latent->V.data[r * rank + j] = vt[j + r * minmn];
        }
    }

    free(work);
    free(means);
    free(sds);
    free(singular);
    free(u);
    free(vt);
    return BRLMM_OK;
}

static int brlmm_prepare_latents_from_K(const BrlmmMatrix *K,
                                        double inertia_threshold,
                                        BrlmmLatent *latent) {
    size_t n = K->rows;
    double *colK = NULL;
    if (brlmm_alloc_col_major(K->data, n, n, &colK) != BRLMM_OK) {
        return BRLMM_ERR_MEMORY;
    }
    double *eigenvalues = (double *)calloc(n, sizeof(double));
    if (!eigenvalues) {
        free(colK);
        return BRLMM_ERR_MEMORY;
    }
    lapack_int n_int = (lapack_int)n;
    lapack_int info = 0;
#if BRLMM_HAVE_LAPACKE
    info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U',
                         n_int, colK, n_int, eigenvalues);
#else
    lapack_int lwork = -1;
    double wkopt;
    dsyev_("V", "U", &n_int, colK, &n_int, eigenvalues,
           &wkopt, &lwork, &info);
    if (info == 0) {
        lwork = (lapack_int)wkopt;
        double *work_dsyev = (double *)malloc((size_t)lwork * sizeof(double));
        if (!work_dsyev) {
            free(colK);
            free(eigenvalues);
            return BRLMM_ERR_MEMORY;
        }
        dsyev_("V", "U", &n_int, colK, &n_int, eigenvalues,
               work_dsyev, &lwork, &info);
        free(work_dsyev);
    }
#endif
    if (info != 0) {
        free(colK);
        free(eigenvalues);
        return BRLMM_ERR_UNSUPPORTED;
    }

    for (size_t i = 0; i < n; ++i) {
        if (eigenvalues[i] < 0.0) {
            eigenvalues[i] = 0.0;
        }
    }
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
        total += eigenvalues[i];
    }
    if (total <= 0.0) {
        total = 1.0;
    }
    double cumulative = 0.0;
    size_t rank = 0;
    double threshold = fmax(fmin(inertia_threshold, 0.999999), 0.0);
    for (size_t i = 0; i < n; ++i) {
        size_t idx = n - 1 - i;
        cumulative += eigenvalues[idx] / total;
        if (cumulative >= threshold) {
            rank = i + 1;
            break;
        }
    }
    if (rank == 0) {
        rank = (n > 0) ? 1 : 0;
    }

    int rc = brlmm_allocate_matrix(&latent->U, n, rank);
    if (rc != BRLMM_OK) {
        free(colK);
        free(eigenvalues);
        return rc;
    }
    rc = brlmm_allocate_vector(&latent->scaled, rank);
    if (rc != BRLMM_OK) {
        brlmm_latent_clear(latent);
        free(colK);
        free(eigenvalues);
        return rc;
    }
    rc = brlmm_allocate_vector(&latent->eigen, rank);
    if (rc != BRLMM_OK) {
        brlmm_latent_clear(latent);
        free(colK);
        free(eigenvalues);
        return rc;
    }
    rc = brlmm_allocate_vector(&latent->singular, rank);
    if (rc != BRLMM_OK) {
        brlmm_latent_clear(latent);
        free(colK);
        free(eigenvalues);
        return rc;
    }
    latent->kind = BRLMM_LATENT_RANDOM_EFFECT;
    latent->rank = rank;

    double sum_selected = 0.0;
    for (size_t j = 0; j < rank; ++j) {
        size_t src = n - 1 - j;
        double eval = eigenvalues[src];
        latent->eigen.data[j] = eval;
        latent->singular.data[j] = sqrt(eval);
        sum_selected += eval;
    }
    if (sum_selected <= 0.0) {
        sum_selected = 1.0;
    }
    for (size_t j = 0; j < rank; ++j) {
        latent->scaled.data[j] = ((double)n * latent->eigen.data[j]) / sum_selected;
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < rank; ++j) {
            size_t src = n - 1 - j;
            latent->U.data[i * rank + j] = colK[i + src * n];
        }
    }

    free(colK);
    free(eigenvalues);
    return BRLMM_OK;
}
