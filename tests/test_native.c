#include "../src/brlmm.h"
#include "../src/brlmm_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct {
    unsigned long state;
} SimpleRng;

double rng_normal(void *ctx, double mean, double sd) {
    SimpleRng *rng = (SimpleRng *)ctx;
    rng->state = rng->state * 6364136223846793005ULL + 1ULL;
    double u1 = ((rng->state >> 11) & 0x1FFFFFULL) / (double)(1ULL << 21);
    rng->state = rng->state * 6364136223846793005ULL + 1ULL;
    double u2 = ((rng->state >> 11) & 0x1FFFFFULL) / (double)(1ULL << 21);
    double r = sqrt(-2.0 * log(fmax(u1, 1e-12)));
    double theta = 2.0 * M_PI * u2;
    return mean + sd * r * cos(theta);
}

double rng_gamma(void *ctx, double shape, double rate) {
    SimpleRng *rng = (SimpleRng *)ctx;
    double sum = 0.0;
    int k = (int)ceil(shape);
    double d = shape - (double)k;
    for (int i = 0; i < k; ++i) {
        rng->state = rng->state * 6364136223846793005ULL + 1ULL;
        double u = ((rng->state >> 11) & 0x1FFFFFULL) / (double)(1ULL << 21);
        sum += -log(fmax(u, 1e-12));
    }
    return (sum + d) / rate;
}

double rng_uniform(void *ctx, double min, double max) {
    SimpleRng *rng = (SimpleRng *)ctx;
    rng->state = rng->state * 6364136223846793005ULL + 1ULL;
    double u = ((rng->state >> 11) & 0x1FFFFFULL) / (double)(1ULL << 21);
    return min + (max - min) * u;
}

int main(void) {
    size_t n = 20;
    double y_data[20];
    for (size_t i = 0; i < n; ++i) {
        y_data[i] = sin((double)i);
    }

    BrlmmMatrix X;
    brlmm_allocate_matrix(&X, n, 2);
    for (size_t i = 0; i < n; ++i) {
        X.data[i * 2 + 0] = 1.0;
        X.data[i * 2 + 1] = (double)i / (double)n;
    }

    BrlmmProblem problem = {0};
    problem.y.data = y_data;
    problem.y.size = n;
    problem.X_list.count = 1;
    problem.X_list.items = &X;

    BrlmmConfig config = {0};
    config.n_iter = 2000;
    config.burnin = 500;
    config.thinning = 10;
    config.method = BRLMM_METHOD_HORSESHOE;
    config.inertia_threshold = 0.95;
    config.scale_X = false;

    SimpleRng rng_state = { (unsigned long)time(NULL) };
    BrlmmRng rng = {
        .normal = rng_normal,
        .gamma = rng_gamma,
        .uniform = rng_uniform,
        .state = &rng_state
    };

    BrlmmOutput output = {0};
    int rc = brlmm_run(&problem, &config, &rng, &output);
    if (rc != BRLMM_OK) {
        fprintf(stderr, "brlmm_run failed: %d\n", rc);
        return EXIT_FAILURE;
    }

    printf("Samples: %zu\n", output.sample_count);
    printf("Mean mu: %.6f\n", output.mu_mean);

    brlmm_output_clear(&output);
    brlmm_free_matrix(&X);
    return EXIT_SUCCESS;
}
