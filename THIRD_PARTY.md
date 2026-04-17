# Third-Party Code

This repository vendors source code from the following third-party projects.
Each vendored component is listed with its upstream source, pinned revision,
and the license under which it is redistributed here.

## Argon2 reference implementation

- **Component**: Argon2 password hashing function (RFC 9106), reference C
  implementation.
- **Upstream**: [P-H-C/phc-winner-argon2](https://github.com/P-H-C/phc-winner-argon2)
- **Pinned revision**: `f57e61e19229e23c4445b85494dbf7c07de721cb` (2021-06-25)
- **Upstream license**: Dual-licensed under Creative Commons CC0 1.0
  Universal (public domain dedication) **or** Apache License 2.0, at the
  licensee's option.
- **License used in this repository**: **Creative Commons CC0 1.0 Universal**.
- **Files vendored** (under `Sources/CArgon2/`):
  - `argon2.c`, `core.c`, `core.h`, `ref.c`, `thread.c`, `thread.h`,
    `encoding.c`, `encoding.h`
  - `include/argon2.h`
  - `blake2/blake2b.c`, `blake2/blake2.h`, `blake2/blake2-impl.h`,
    `blake2/blamka-round-ref.h`
  - `LICENSE` (verbatim upstream license file)
- **Modifications**: None. Files are bit-identical to the pinned upstream
  revision. The Swift Package Manager target `CArgon2` selects the portable
  reference round function (`ref.c`); the x86-specific `opt.c` and build
  harness files (`bench.c`, `run.c`, `test.c`, `genkat.c`) are not included.
- **Rationale**: Vendoring the reference implementation directly eliminates
  any dependency on a third-party Swift wrapper. The Argon2 algorithm is a
  finalized standard (RFC 9106); the reference C source is authored and
  maintained by the algorithm's co-authors and is the canonical implementation.

## Attribution

The CC0 1.0 Universal dedication waives all copyright and related rights in
the vendored source code to the extent possible under law. No attribution is
legally required, but the original authors (Daniel Dinu, Dmitry Khovratovich,
Jean-Philippe Aumasson, Samuel Neves) are credited here in recognition of
their work on the Password Hash Competition and the Argon2 algorithm.
