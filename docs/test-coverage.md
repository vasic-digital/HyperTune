# Test Coverage — digital.vasic.hypertune (round-256)

> Verbatim 2026-05-19 operator mandate: *"all existing tests and Challenges do work in anti-bluff manner - they MUST confirm that all tested codebase really works as expected! We had been in position that all tests do execute with success and all Challenges as well, but in reality the most of the features does not work and can't be used! This MUST NOT be the case and execution of tests and Challenges MUST guarantee the quality, the completition and full usability by end users of the product!"*

CONST-050(B) symbol-to-test ledger. Every exported symbol in `pkg/{client,types}` is cross-referenced to the test name(s) that exercise it AND to the round-256 Challenge runner section that exercises it through real public API calls with bilingual (en/sr/ja/ar/zh-CN) prompts and an injected deterministic LLM surrogate. No metadata-only PASS — every entry below names the production code path and the runtime evidence channel that proves it works.

## Anti-bluff posture (round-256)

- **Real public API exercise.** `challenges/runner/main.go` invokes the same `hypertune.New() / SetRunner / RegisterMetric / SetSeed / Optimize / GridSearch / BayesianOptimize / Evaluate / GetMetrics / SuggestParameters` API surface a downstream consumer (HelixAgent's LLMProvider, an Ollama HTTP client, an Anthropic SDK call) would use. HyperTune itself is the System Under Test — only the LLM behind the Runner injection seam is replaced by a deterministic in-process surrogate exactly as the seam was designed to permit.
- **Bilingual prompt-byte preservation.** Section 2 runs 5 locales × 3 search methods = 15 Optimize() invocations and asserts (a) the Runner observed prompt bytes equal the bytes passed in, and (b) every trial output retains the prompt as a prefix. Drift on any locale (Latin, Cyrillic, CJK, Arabic, Chinese) is a hard FAIL.
- **Round-23 §11.4 sentinel preserved.** Section 8 plants a fresh client WITHOUT `SetRunner` and asserts EVERY entry point (`Optimize`, `GridSearch`, `BayesianOptimize`, `Evaluate`) returns `ErrBaselineRunnerNotConfigured` via `errors.Is`. This is the round-23 audit fix that previously caused the library to fabricate optimisation data; round-256 makes its preservation a permanent gate.
- **Grid dimension invariant.** Section 3 asserts `GridSearch` produces exactly 12 points (4 temps × 3 topPs) — same invariant as `TestGridSearchProducesNonEmptyGrid` but exercised through the Challenge so a refactor that silently changes grid size is caught by both layers.
- **Bayesian phase coverage.** Section 4 invokes `BayesianOptimize` with both a short budget (iters=2, exercises seed-only path) and a learning budget (iters=6, exercises 3-random-seed + 3-perturbation phases).
- **Metric injection end-to-end.** Section 6 asserts `GetMetrics` surfaces both the 3 builtins (`default`, `length`, `exact_match`) and the custom `bilingual_runelen` metric registered via `RegisterMetric`.
- **Perturbation budget.** Section 7 asserts `SuggestParameters` with non-empty history stays within ±0.2 of the best score's temperature — proves the perturbation behaviour is the intended local search rather than a re-sample.
- **Paired mutation.** `hypertune_describe_challenge.sh --anti-bluff-mutate` plants a `BayesianOptimize -> BayesianBogus_MUTATED` rename in a tmp ledger copy and asserts the gate exits 99. Proves the cross-reference gate catches ledger-vs-source drift instead of rubber-stamping it.

## pkg/types

| Exported symbol | Unit-test coverage | Runner section |
|-----------------|--------------------|----------------|
| `type ParameterSpace` | every `TestOptimize*`, `TestGridSearchProducesNonEmptyGrid`, `TestBayesianOptimizeReducesToSeedOnShortBudget` | Section 2, 3, 4 (passed as `ParameterSpace{}` and `ParameterSpace{Temperature, TopP}`) |
| `(*ParameterSpace).Defaults()` | `TestParameterSpaceDefaults` (types_test.go) | Section 9 (`zero.Defaults()` assertions on MaxTokens=2048, Temperature=0.7, TopP=1.0) |
| `type OptimizationConfig` | every `TestOptimize*` | Section 2 (Method=random/grid/bayesian) + Section 9 |
| `(*OptimizationConfig).Validate()` | `TestOptimizationConfigValidateValid`, `TestOptimizationConfigValidateEmpty`, `TestOptimizeInvalidConfig` | Section 9 (empty + populated cases) |
| `type OptimizationResult` | every `TestOptimize*` (returned value) | Section 2 (Iterations, BestScore, BestParams, AllResults inspected per trial) |
| `type TrialResult` | `TestEvaluate`, `TestSuggestParametersPerturbsAroundBest` | Section 2 (AllResults trial inspection), Section 5 (single-trial), Section 7 (history input) |
| `type EvaluationMetric` | `TestEvaluationMetricValidateValid`, `TestEvaluationMetricValidateEmpty`, `TestGetMetrics` | Section 6 (GetMetrics returns []EvaluationMetric) |
| `(*EvaluationMetric).Validate()` | `TestEvaluationMetricValidateValid`, `TestEvaluationMetricValidateEmpty` | Section 9 (`emptyMet.Validate()` rejection) |

## pkg/client

| Exported symbol | Unit-test coverage | Runner section |
|-----------------|--------------------|----------------|
| `type Client` | `TestNew`, `TestDoubleClose`, `TestConfig`, every `TestOptimize*` | Section 1 (construction + lifecycle) |
| `type Runner` (function type) | `dotsTestRunner` in client_test.go is a value of this type | Section 1 (`po.surrogateRunner`) |
| `type Metric` (function type) | `TestRegisterMetricCollision`, `TestRegisterMetricIgnoresNilOrEmpty`, `TestSetRunnerAndRegisterMetric` | Section 1 (`bilingualRuneLengthMetric`) + Section 6 |
| `var ErrBaselineRunnerNotConfigured` | `TestOptimizeWithoutInjectedRunner_ReturnsSentinel`, `TestGridSearchWithoutInjectedRunner_ReturnsSentinel`, `TestBayesianOptimizeWithoutInjectedRunner_ReturnsSentinel`, `TestEvaluateWithoutInjectedRunner_ReturnsSentinel` | Section 8 (all 4 entry points: Optimize, GridSearch, BayesianOptimize, Evaluate) |
| `func New(opts ...config.Option) (*Client, error)` | `TestNew`, `TestDoubleClose`, `TestConfig`, every `TestOptimize*` indirectly | Section 1 (construction) + Section 8 (bare client) |
| `func NewFromConfig(cfg *config.Config) (*Client, error)` | compile-time + `TestConfig` (asserts `Config()` returns non-nil) | indirectly via Section 1 |
| `(*Client).Close() error` | `TestNew`, `TestDoubleClose` (idempotent), `TestConfig` | Section 1 (deferred close), Section 8 (deferred close on bare client) |
| `(*Client).Config() *config.Config` | `TestConfig` | n/a (not exercised by Challenge; covered by unit) |
| `(*Client).SetRunner(r Runner)` | `TestSetRunnerAndRegisterMetric`, `TestSetRunnerNilIgnored`, `TestOptimizeRunnerErrorPropagates`, every `TestOptimize*` via `newTestClient` | Section 1 (installation) — Sections 2-7 exercise the installed surrogate |
| `(*Client).RegisterMetric(name string, m Metric)` | `TestRegisterMetricCollision`, `TestRegisterMetricIgnoresNilOrEmpty`, `TestSetRunnerAndRegisterMetric` | Section 1 (installation) + Section 6 (GetMetrics surfaces custom metric) |
| `(*Client).SetSeed(seed int64)` | `TestSeedReproducibility`, `TestSeedDivergenceWithDifferentSeeds` | Section 1 (initial seed), Section 2 (re-seeded per loop iter), Section 4 (re-seeded per BO budget), Section 7 (seeded before SuggestParameters) |
| `(*Client).Optimize(ctx, space, cfg) (*OptimizationResult, error)` | `TestOptimizeRandom`, `TestOptimizeGrid`, `TestOptimizeBayesian`, `TestOptimizeUnknownMethod`, `TestOptimizeInvalidConfig`, `TestOptimizeRunnerErrorPropagates`, `TestOptimizeWithoutInjectedRunner_ReturnsSentinel`, `TestSetRunnerAndRegisterMetric` | Section 2 (15 invocations: 5 locales × 3 methods) + Section 8 (sentinel) |
| `(*Client).GridSearch(ctx, space, cfg) (*OptimizationResult, error)` | `TestOptimizeGrid` (via dispatch), `TestGridSearchProducesNonEmptyGrid`, `TestGridSearchWithoutInjectedRunner_ReturnsSentinel` | Section 3 (12-point grid invariant) + Section 8 (sentinel) |
| `(*Client).BayesianOptimize(ctx, space, cfg) (*OptimizationResult, error)` | `TestOptimizeBayesian` (via dispatch), `TestBayesianOptimizeReducesToSeedOnShortBudget`, `TestBayesianOptimizeWithoutInjectedRunner_ReturnsSentinel` | Section 4 (iters=2 + iters=6) + Section 8 (sentinel) |
| `(*Client).Evaluate(ctx, params, prompt, model) (*TrialResult, error)` | `TestEvaluate`, `TestEvaluateEmptyPrompt`, `TestEvaluateWithoutInjectedRunner_ReturnsSentinel`, `TestRegisterMetricCollision`, `TestSetRunnerNilIgnored` | Section 5 (per-locale single-trial) + Section 8 (sentinel) |
| `(*Client).GetMetrics(ctx) ([]EvaluationMetric, error)` | `TestGetMetrics`, `TestRegisterMetricIgnoresNilOrEmpty` | Section 6 (builtins + custom surface) |
| `(*Client).SuggestParameters(ctx, space, history) (map[string]float64, error)` | `TestSuggestParameters`, `TestSuggestParametersPerturbsAroundBest` | Section 7 (empty history + 3-entry history + perturbation budget) |

## Verification

```bash
# Unit-test floor (testify) — all packages
GOMAXPROCS=2 nice -n 19 go test -race -count=1 -p 1 ./...

# Round-256 Challenge runner + paired-mutation
bash challenges/hypertune_describe_challenge.sh                       # exit 0
bash challenges/hypertune_describe_challenge.sh --anti-bluff-mutate   # exit 99
```

The paired-mutation invocation is the load-bearing proof that the cross-reference gate (Section 2 of `hypertune_describe_challenge.sh`) catches ledger-vs-source drift — a ledger that silently lists nothing would PASS Section 2 vacuously without the mutation check. Exit 99 means the gate FAILED on the planted mutation, which is the desired anti-bluff behaviour.
