# HyperTune

> **SCAFFOLD / WORK IN PROGRESS -- NOT YET FUNCTIONAL**
>
> This module publishes a compiling Go scaffold with `ErrCodeUnimplemented`
> returned from every method body. Phase-A implementation (~4 days core
> surface per module) is a future milestone.

## Purpose

Hyperparameter tuning orchestration. Once Phase-A is authorised and implemented the module will
provide the above capability as a consumable Go library for the
HelixAgent ensemble and its siblings.

## Status

- Compiles: `go build ./...` exits 0 when sibling scaffold repos are
  checked out alongside (see Development layout below).
- Method bodies: all return `ErrCodeUnimplemented`.
- Integration: no runtime integration with consumers.
- Future: see the Phase-A plan in the consuming HelixAgent repo at
  `docs/superpowers/specs/2026-04-21-elder-plinius-phaseA-go-hypertune.md`.

## Lineage

Extracted from internal HelixAgent research tree on 2026-04-21. The
earlier Python upstream name was obfuscated (leetspeak); this Go port
uses a clean readable name.

Original research corpus: `docs/research/go-elder-plinius-v3/go-elder-plinius/go-hypertune/`
inside the HelixAgent repository.

## Module path

```go
import "digital.vasic.hypertune"
```

## Development layout

This scaffold's `go.mod` declares the module as `digital.vasic.hypertune` and
(where applicable) uses relative `replace` directives such as
`../PliniusCommon` to consume sibling scaffolds. To build locally,
clone the sibling repos next to this one:

```
workspace/
  PliniusCommon/
  HyperTune/
  ... other siblings ...
```

HelixAgent consumers pin these modules via their own `replace`
directives pointing at the appropriate submodule path.

## License

Apache-2.0
