#ifndef BRLMM_H
#define BRLMM_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Public interface for the BRLMM sampler port to C.
 * The API mirrors the R implementation in brlmm.R and
 * is intended to support the Horseshoe variant of the sampler.
 */

#define BRLMM_VERSION_MAJOR 0
#define BRLMM_VERSION_MINOR 1
#define BRLMM_VERSION_PATCH 0

/* Matrix-valued inputs may be stored row-major or column-major.
 * All helper routines in this header assume row-major storage and
 * contiguous buffers. Client code is responsible for allocating
 * and releasing any memory referenced by these structs unless
 * explicitly documented otherwise.
 */
/*
 * Data layout contract
 * ---------------------
 * All `BrlmmMatrix` instances consumed by the C core MUST expose data in
 * row-major order (i.e. consecutive entries correspond to advancing the
 * column index). Callers are responsible for arranging their inputs to
 * satisfy this contract; the implementation treats the buffer as
 * row-major without further checks. Violating this assumption leads to
 * incorrect linear algebra computations.
 */

typedef enum {
    BRLMM_METHOD_HORSESHOE = 0
} BrlmmMethod;

typedef enum {
    BRLMM_LATENT_FIXED_EFFECT = 0,
    BRLMM_LATENT_RANDOM_EFFECT = 1
} BrlmmLatentKind;

typedef enum {
    BRLMM_OK = 0,
    BRLMM_ERR_INVALID_ARGUMENT = -1,
    BRLMM_ERR_DIMENSION_MISMATCH = -2,
    BRLMM_ERR_MEMORY = -3,
    BRLMM_ERR_UNSUPPORTED = -4
} BrlmmErrorCode;

typedef struct {
    size_t size;
    double *data;
} BrlmmVector;

typedef struct {
    size_t rows;
    size_t cols;
    double *data; /* row-major, length rows * cols */
} BrlmmMatrix;

typedef struct {
    BrlmmLatentKind kind;
    size_t rank;          /* number of latent factors retained */
    BrlmmMatrix U;        /* n x rank */
    BrlmmMatrix V;        /* p x rank (fixed effects only, otherwise cols = 0) */
    BrlmmVector singular; /* length rank, corresponds to D in brlmm.R */
    BrlmmVector eigen;    /* length rank, corresponds to D2 for random effects */
    BrlmmVector scaled;   /* length rank, corresponds to sD2 */
    BrlmmVector centers;  /* length p, fixed effects only */
    BrlmmVector scale;    /* length p, fixed effects only */
} BrlmmLatent;

typedef struct {
    double (*normal)(void *state, double mean, double sd);
    double (*gamma)(void *state, double shape, double rate);
    double (*uniform)(void *state, double min, double max);
    void *state;
} BrlmmRng;

typedef struct {
    size_t count;
    const BrlmmMatrix *items;
} BrlmmMatrixList;

typedef struct {
    BrlmmVector y;           /* length n */
    BrlmmMatrixList X_list;  /* fixed effect design matrices */
    BrlmmMatrixList K_list;  /* random effect kernels */
} BrlmmProblem;

typedef struct {
    size_t n_iter;
    size_t burnin;
    size_t thinning;
    double inertia_threshold;
    bool scale_X;
    BrlmmMethod method;
} BrlmmConfig;

typedef struct {
    size_t count;
    BrlmmLatent *items;
} BrlmmLatentArray;

typedef struct {
    size_t n;               /* number of observations */
    BrlmmVector y;          /* copy or view of response vector */
    BrlmmLatentArray latents;
    size_t LX;              /* number of fixed-effect latent blocks */
    size_t LK;              /* number of random-effect latent blocks */
} BrlmmInput;

typedef struct {
    size_t count;
    BrlmmVector *items;
} BrlmmVectorArray;

typedef struct {
    double mu;
    double sigma2;
    double tau2;
    double xi2_tau2;
    BrlmmVector residuals;      /* length n */
    BrlmmVector lambda;         /* length L */
    BrlmmVector omega2;         /* length L, Horseshoe only */
    BrlmmVector xi2_omega2;     /* length L, Horseshoe only */
    BrlmmVectorArray nu;        /* per-latent scores, each length rank */
} BrlmmState;

typedef struct {
    double mu;
    double sigma2;
    double loss;
    BrlmmVector residuals;  /* length n */
    BrlmmVector lambda;     /* length L */
    BrlmmVectorArray nu;    /* per-latent scores */
} BrlmmSample;

typedef struct {
    size_t sample_count;
    size_t L;
    size_t LX;
    size_t LK;
    size_t n;                 /* original response length */
    BrlmmVector mu_chain;     /* length sample_count */
    BrlmmVector sigma2_chain; /* length sample_count */
    BrlmmMatrix lambda_chain; /* shape L x sample_count */
    BrlmmVectorArray nu_chain;/* per latent: rank x sample_count matrices */
    BrlmmVector mse_chain;    /* length sample_count */
    BrlmmVector loss_chain;   /* length sample_count */
    BrlmmLatentArray latents; /* copy of latent descriptors */
    double mu_mean;
} BrlmmOutput;

/* High-level pipeline ---------------------------------------------------- */

int brlmm_parse_input(const BrlmmProblem *problem,
                      const BrlmmConfig *config,
                      BrlmmInput *out_input);

void brlmm_input_clear(BrlmmInput *input);

int brlmm_state_init(const BrlmmInput *input,
                     BrlmmMethod method,
                     BrlmmState *out_state);

void brlmm_state_clear(BrlmmState *state);

int brlmm_update_state_horseshoe(BrlmmState *state,
                                 const BrlmmLatentArray *latents,
                                 const BrlmmRng *rng);

int brlmm_compute_loss(const BrlmmInput *input,
                       const BrlmmState *state,
                       BrlmmMethod method,
                       double *out_loss);

int brlmm_record_sample(const BrlmmInput *input,
                        const BrlmmState *state,
                        BrlmmSample *out_sample);

void brlmm_sample_clear(BrlmmSample *sample);

int brlmm_run(const BrlmmProblem *problem,
              const BrlmmConfig *config,
              const BrlmmRng *rng,
              BrlmmOutput *out_result);

void brlmm_output_clear(BrlmmOutput *output);

int brlmm_predict(const BrlmmOutput *fit,
                  const BrlmmMatrixList *new_X_list,
                  const BrlmmMatrixList *new_K_list,
                  BrlmmMatrix *out_chain);

/* Latent construction helpers ------------------------------------------- */

int brlmm_compute_latents_from_X(const BrlmmMatrixList *X_list,
                                 double inertia_threshold,
                                 bool scale_X,
                                 size_t n,
                                 BrlmmLatentArray *out_latents);

int brlmm_compute_latents_from_K(const BrlmmMatrixList *K_list,
                                 double inertia_threshold,
                                 size_t n,
                                 BrlmmLatentArray *out_latents);

#ifdef __cplusplus
}
#endif

#endif /* BRLMM_H */
