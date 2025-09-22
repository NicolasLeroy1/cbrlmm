# CBRLMM: C Bayesian Regularization of Linear Mixed Models

CBRLMM is a portable C implementation of a **Bayesian Regularization of Linear Mixed Models** sampler. It mirrors the reference `brlmm.R` implementation while
exposing a fast native library that other languages can call through FFI:

- a pure C library (`cbrlmm`) that performs data preparation, latent-factor
  decomposition, and Horseshoe-regularized Gibbs sampling, and
- language bridges (e.g. the R bridge under `bridge/`) plus bindable headers so
  other runtimes can call the sampler via FFI.

## Why and When to Use BRLMM

Reach for BRLMM when:

- You work with heterogeneous feature sets (many categorical/continuous blocks)
  and want the model to infer which latent groups explain the response.
- You need to combine several kernel constructions—BRLMM can ingest multiple
  random-effect kernels and lets the sampler retain the most relevant ones.
- Your data are high-dimensional (*p* ≫ *n*) or strongly correlated (e.g.
  genomic panels) where the Horseshoe prior and latent low-rank structure shine.

## Model Overview

Given a response vector `y` (length *n*), the BRLMM pipeline accepts optional
fixed-effect design matrices (`X_list`) and random-effect kernel matrices
(`K_list`).  Each matrix is decomposed into a low-rank latent representation via
SVD/eigendecomposition, retaining enough components to explain a specified
fraction of inertia.  During sampling the model learns effect scores (`nu`),
loadings (`lambda`), and variance components using a Horseshoe prior that
encourages sparsity across latent factors.

The R reference implementation (`brlmm.R`) proceeds as follows:

1. **Input parsing** – validate shapes, convert single matrices to lists, and
   perform SVD/eigen decompositions to build latent blocks.
2. **Initialisation** – center the response, seed residuals, and set starting
   values for variance hyperparameters and latent scores.
3. **Gibbs sampling loop** – for each iteration update `nu`, `lambda`, the
   Horseshoe local scales and global variance (`tau²`, `sigma²`), capturing
   draws after burn-in/thinning.
4. **Output** – aggregate posterior draws into chains (mu, sigma², lambda,
   per-latent scores), compute losses, and provide prediction helpers.

## Library Layout

```
cbrlmm/
├── src/               # R-free core (public headers + implementation)
│   ├── brlmm.c/.h
│   ├── brlmm_utils.c/.h
│   ├── Makefile        # builds libbrlmm_core.{a,so} and test_native
│   └── README.md       # build and linkage instructions
├── bridge/            # Language bindings (currently R)
│   └── brlmm_bridge.c
├── benchmark/         # Simulation and benchmarking scripts (R)
├── tests/             # R comparison + native C smoke test
└── README.md          # This document
```

## Building the Core Library (`cbrlmm`)

```
cd src
make            # produces libbrlmm_core.a, libbrlmm_core.so, and test_native
./test_native   # run the C smoke test
```

The `Makefile` links against BLAS/LAPACK; adjust via `LIBS` if you prefer
OpenBLAS, MKL, etc. Public headers (`brlmm.h`, `brlmm_utils.h`) live in `src/`
and expose the sampling API (`brlmm_run`, `brlmm_predict`, etc.).

## Using the R Bridge

From the project root:

```
Rscript tests/test_compare.R   # compiles bridge/brlmm_bridge.c and runs parity test
```

The bridge builds a shared library (`bridge/brlmm_port.so`) that the R functions
`run_brlmm_c()` consume.  The two implementations share identical outputs, so
existing R code can swap between native C and pure R samplers.

## Bindings & FFI

Other languages can link against `libbrlmm_core` by including `brlmm.h`,
constructing `BrlmmProblem`, `BrlmmConfig`, and `BrlmmRng` structs, then calling
`brlmm_run`. See `tests/test_native.c` for a minimal example.

## Benchmarks

`benchmark/benchmark.R` runs cross-validation experiments comparing the R and C
samplers on simulated data. Performance gains increase with the number of latent
blocks (high LX/LK), because latent updates dominate runtime.

## When to use the C implementations over the R implementation

- **Large latent structures** – When your model carries many fixed or random
  effect blocks (hundreds of SVD/eigen decompositions), the C sampler avoids the
  interpreter overhead that dominates the pure-R implementation, often cutting
  runtime by 2× or more.
- **Cross-language integration** – If you need BRLMM inside a non-R runtime
  (Python, Rust, C++, etc.), the standalone library exposes a stable C ABI that
  you can bind with standard FFI tooling and ship as part of your own services
  or pipelines.
- **Production deployments** – Embedding the native sampler in long-running
  jobs or services is easier when you can link to a shared object rather than
  spinning up an R session per request.
- **Research prototyping** – The native sampler mirrors the reference R
  behaviour, so you can prototype in R, then switch to the C engine for faster
  sweeps or larger simulation studies without rewriting model logic.

Stick with the R implementation when you need rapid experimentation entirely in
R, or when the datasets are small enough that the native speedup is negligible.

---

CBRLMM brings the flexibility of the original R implementation to any runtime
with an FFI. Contributions—new bindings, model extensions, or performance
optimisations—are welcome.
