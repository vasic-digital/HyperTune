# HyperTune

Automated hyperparameter optimisation for LLM inference: `temperature`,
`top_p`, `top_k`, `repetition_penalty`, `presence_penalty`,
`frequency_penalty`, and `max_tokens` — via random search, grid search,
or a Bayesian-optimisation-lite variant. Part of the Plinius Go service
family used by HelixAgent.

## Status

- Compiles: `go build ./...` exits 0.
- Tests pass under `-race`: 2 packages (`types`, `client`), all green.
- Baseline default Runner returns `ErrBaselineRunnerNotConfigured` —
  callers MUST inject a real LLM-dispatching Runner via `SetRunner`
  before invoking `Optimize` / `GridSearch` / `BayesianOptimize` /
  `Evaluate` (round-23 §11.4 audit fix; see CONST-035 anti-bluff).
- Default metrics (`default`, `length`, `exact_match`) seeded on
  `New()`; register custom metrics via `RegisterMetric`.
- Integration-ready: consumable Go library for the HelixAgent ensemble.

## Purpose

- `pkg/types` — value types: `ParameterSpace`, `OptimizationConfig`,
  `OptimizationResult`, `TrialResult`, `EvaluationMetric`.
- `pkg/client` — parameter search orchestration:
  - `Optimize(space, cfg)` — dispatches on `cfg.Method`
  - `GridSearch`, `BayesianOptimize`, random-search baseline
  - `Evaluate(params, prompt, model)` — single-trial scoring
  - `GetMetrics`, `SuggestParameters(space, history)`
  - `SetRunner(Runner)` / `RegisterMetric(name, Metric)` / `SetSeed`

## Usage

```go
import (
    "context"
    "log"
    "time"

    hypertune "digital.vasic.hypertune/pkg/client"
    "digital.vasic.hypertune/pkg/types"
)

c, err := hypertune.New()
if err != nil { log.Fatal(err) }
defer c.Close()

// REQUIRED — inject a real LLM-dispatching Runner. Without this every
// optimisation call returns ErrBaselineRunnerNotConfigured (round-23 §11.4 fix).
c.SetRunner(func(ctx context.Context, prompt string, params map[string]float64) (string, time.Duration, error) {
    // real call to OpenAI, Anthropic, Ollama, HelixAgent's LLMProvider, etc.
    return callMyLLM(ctx, prompt, params)
})

res, err := c.Optimize(context.Background(), types.ParameterSpace{}, types.OptimizationConfig{
    Model:      "gpt-4",
    Prompt:     "Summarise this article in 3 bullets.",
    Method:     "random",
    Iterations: 20,
})
if err != nil { log.Fatal(err) }
log.Printf("best params: %+v (score=%.3f)", res.BestParams, res.BestScore)
```

## Anti-bluff guarantees (round-256)

The library + its Challenge suite together carry the following load-bearing
anti-bluff invariants. Every PASS in this codebase MUST satisfy ALL of them
per Article XI §11.9 / CONST-035 / CONST-050(B):

- **Default Runner is a sentinel, not a fabricator.** `New()` installs a
  baseline Runner that returns `ErrBaselineRunnerNotConfigured`. Callers MUST
  inject a real Runner via `SetRunner` before any optimisation call. The
  previous default fabricated outputs and scored them with `defaultMetrics`,
  producing meaningless "best parameters" reports — round-23 §11.4 PASS-bluff
  at the library-default layer (resolved 2026-05-17; commit `1237b9f`).
- **Sentinel propagates through every entry point.** `Optimize`,
  `GridSearch`, `BayesianOptimize`, and `Evaluate` all surface
  `ErrBaselineRunnerNotConfigured` (wrapped via `errors.Wrap`, unwrappable
  via `errors.Is`) when no real Runner has been injected. Covered by 4 unit
  tests (`Test*WithoutInjectedRunner_ReturnsSentinel`) + Section 8 of the
  round-256 Challenge runner.
- **Bilingual prompt bytes round-trip verbatim.** The round-256 Challenge
  runner drives Optimize × 3 methods × 5 locales (en/sr/ja/ar/zh-CN) and
  asserts the Runner observed prompt bytes equal the bytes passed in, AND
  every trial output retains the prompt as a prefix. Drift on any locale is
  a hard FAIL.
- **GridSearch dimension is deterministic.** `buildGrid` produces exactly
  12 points (4 temps × 3 topPs) for the default ParameterSpace; assertable
  via `TestGridSearchProducesNonEmptyGrid` + Challenge Section 3.
- **Seed reproducibility is contractual.** Identical seed → identical
  `BestParams`; covered by `TestSeedReproducibility` +
  `TestSeedDivergenceWithDifferentSeeds`. The round-256 runner re-seeds
  per loop iteration so its output is byte-identical across runs.
- **Symbol → test ledger.** Every exported symbol in `pkg/{client,types}`
  is cross-referenced in `docs/test-coverage.md` to the test name(s) that
  exercise it AND to the Challenge runner section that exercises it. A
  paired-mutation gate (`challenges/hypertune_describe_challenge.sh
  --anti-bluff-mutate`) plants a deliberate symbol rename in a tmp ledger
  copy and asserts exit 99 — proves the cross-reference gate isn't
  rubber-stamping.

## Module path

```go
import "digital.vasic.hypertune"
```

## Verification

```bash
# Unit-test floor (testify) — all packages
GOMAXPROCS=2 nice -n 19 go test -race -count=1 -p 1 ./...

# Round-256 deep-doc + bilingual Challenge runner
bash challenges/hypertune_describe_challenge.sh                       # exit 0
bash challenges/hypertune_describe_challenge.sh --anti-bluff-mutate   # exit 99
```

## Lineage

Extracted from internal HelixAgent research tree on 2026-04-21.
Graduated to functional status on the same day alongside its 7 sibling
Plinius modules. Round-23 §11.4 audit (2026-05-17) removed the
fabricating baseline Runner; round-256 (2026-05-19) added the bilingual
deep-doc Challenge + paired-mutation gate.

Historical research corpus (unused) remains at
`docs/research/go-elder-plinius-v3/go-elder-plinius/go-hypertune/`
inside the HelixAgent repository.

## Development layout

This module's `go.mod` declares the module as `digital.vasic.hypertune`
and uses a relative `replace` directive pointing at `../PliniusCommon`.

## License

Apache-2.0
