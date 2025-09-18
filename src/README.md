# BRLMM Core Library

This directory now contains only the R-independent C implementation of the
BRLMM sampler.  You can build a static and shared library with:

```sh
make
```

This produces `libbrlmm_core.a` and `libbrlmm_core.so` which export the public
API declared in `brlmm.h`.  The build expects a BLAS/LAPACK toolchain to be
available (link flags default to `-llapack -lblas`); you can override
`LIBS`/`CFLAGS` via environment variables, for example:

```sh
make LIBS="-lopenblas"
```

The R bridge has moved to `bridge/brlmm_bridge.c`.  When compiling that module,
point your include path at this directory, e.g.

```sh
R CMD SHLIB -I../src bridge/brlmm_bridge.c ../src/brlmm.c ../src/brlmm_utils.c
```
