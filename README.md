# HyperTune

Automated hyperparameter optimisation for LLM inference: temperature,
top_p, top_k, repetition_penalty, presence_penalty, frequency_penalty,
and max_tokens — via random search, grid search, or Bayesian-lite
optimisation. Part of the Plinius Go service family used by HelixAgent.

## Status

- Compiles: `go build ./...` exits 0.
- Tests pass under `-race`: 2 packages (types, client), all green.
- Baseline deterministic runner + default metrics seeded so the client
  is immediately usable in tests; inject a real LLM runner and metrics
  via `SetRunner` / `RegisterMetric` for production use.
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

    hypertune "digital.vasic.hypertune/pkg/client"
    "digital.vasic.hypertune/pkg/types"
)

c, err := hypertune.New()
if err != nil { log.Fatal(err) }
defer c.Close()

res, err := c.Optimize(context.Background(), types.ParameterSpace{}, types.OptimizationConfig{
    Model:      "gpt-4",
    Prompt:     "Summarise this article in 3 bullets.",
    Method:     "random",
    Iterations: 20,
})
if err != nil { log.Fatal(err) }
log.Printf("best params: %+v (score=%.3f)", res.BestParams, res.BestScore)
```

## Module path

```go
import "digital.vasic.hypertune"
```

## Lineage

Extracted from internal HelixAgent research tree on 2026-04-21.
Graduated to functional status on the same day alongside its 7 sibling
Plinius modules.

Historical research corpus (unused) remains at
`docs/research/go-elder-plinius-v3/go-elder-plinius/go-hypertune/`
inside the HelixAgent repository.

## Development layout

This module's `go.mod` declares the module as `digital.vasic.hypertune`
and uses a relative `replace` directive pointing at `../PliniusCommon`.

## License

Apache-2.0
