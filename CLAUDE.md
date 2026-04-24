# CLAUDE.md -- digital.vasic.hypertune


## Definition of Done

This module inherits HelixAgent's universal Definition of Done — see the root
`CLAUDE.md` and `docs/development/definition-of-done.md`. In one line: **no
task is done without pasted output from a real run of the real system in the
same session as the change.** Coverage and green suites are not evidence.

### Acceptance demo for this module

```bash
# Random + grid + BO-lite optimization with Runner + Metrics injection
cd HyperTune && GOMAXPROCS=2 nice -n 19 go test -count=1 -race -v ./pkg/client
```
Expect: PASS; all three search backends produce `OptimizationResult.BestParams`; registered metrics used for scoring.


Module-specific guidance for Claude Code.

## Status

**FUNCTIONAL.** 2 packages (types, client) ship tested implementations;
`go test -race ./...` all green. Baseline runner + default metrics
(default / length / exact_match) seeded on `New()`. Three search
backends are implemented: random search (default), grid search, and
BO-lite.

## Hard rules

1. **NO CI/CD pipelines** -- no `.github/workflows/`, `.gitlab-ci.yml`,
   `Jenkinsfile`, `.travis.yml`, `.circleci/`, or any automated
   pipeline. No Git hooks either. Permanent.
2. **SSH-only for Git** -- `git@github.com:...` / `git@gitlab.com:...`.
3. **Conventional Commits** -- `feat(hypertune): ...`, `fix(...)`,
   `docs(...)`, `test(...)`, `refactor(...)`.
4. **Code style** -- `gofmt`, `goimports`, 100-char line ceiling,
   errors always checked and wrapped (`fmt.Errorf("...: %w", err)`).
5. **Resource cap for tests** --
   `GOMAXPROCS=2 nice -n 19 ionice -c 3 go test -count=1 -p 1 -race ./...`

## Purpose

Hyperparameter tuning orchestration for LLM inference. Key surface:
`Optimize`, `GridSearch`, `BayesianOptimize`, `Evaluate`, `GetMetrics`,
`SuggestParameters`, `SetRunner`, `RegisterMetric`.

## Primary consumer

HelixAgent (`dev.helix.agent`) — ensemble inference tuning.

## Testing

```
GOMAXPROCS=2 nice -n 19 ionice -c 3 go test -count=1 -p 1 -race ./...
```

## API Cheat Sheet

**Module path:** `digital.vasic.hypertune`.

```go
type Runner func(ctx, prompt string, params map[string]float64) (string, time.Duration, error)
type Metric func(ctx, output, reference string) (float64, error)

type ParameterSpace struct {
    Temperature, TopP, TopK              [2]float64  // [min, max]
    RepetitionPenalty, PresencePenalty   [2]float64
    FrequencyPenalty                     [2]float64
    MaxTokens                            [2]int
}
type OptimizationConfig struct {
    Method     string  // "random" | "grid" | "bayesian"
    Metric     string
    Iterations int
}
type OptimizationResult struct {
    BestParams map[string]float64
    BestScore  float64
    Trials     []TrialResult
}

type Client struct { /* 3 search strategies */ }

func New(opts ...config.Option) (*Client, error)
func (c *Client) SetRunner(r Runner)
func (c *Client) RegisterMetric(name string, m Metric)
func (c *Client) SetSeed(seed int64)
func (c *Client) Optimize(ctx, space ParameterSpace, cfg OptimizationConfig) (*OptimizationResult, error)
func (c *Client) GridSearch(ctx, space, cfg) (*OptimizationResult, error)
func (c *Client) BayesianOptimize(ctx, space, cfg) (*OptimizationResult, error)
func (c *Client) Evaluate(ctx, params map[string]float64, prompt, model string) (*TrialResult, error)
func (c *Client) Close() error
```

**Typical usage:**
```go
c, _ := hypertune.New()
defer c.Close()
c.SetRunner(myLLMRunner)
c.RegisterMetric("bleu", bleuMetric)
r, _ := c.Optimize(ctx,
    hypertune.ParameterSpace{Temperature: [2]float64{0, 2}, TopP: [2]float64{0, 1}},
    hypertune.OptimizationConfig{Method: "bayesian", Metric: "bleu", Iterations: 20})
```

**Injection points:** `Runner`, `Metric`.
**Defaults on `New`:** baseline runner, 3 built-in metrics (`default`, `length`, `exact_match`), random seed.

## Integration Seams

| Direction | Sibling modules |
|-----------|-----------------|
| Upstream (this module imports) | PliniusCommon |
| Downstream (these import this module) | root only |

*Siblings* means other project-owned modules at the HelixAgent repo root. The root HelixAgent app and external systems are not listed here — the list above is intentionally scoped to module-to-module seams, because drift *between* sibling modules is where the "tests pass, product broken" class of bug most often lives. See root `CLAUDE.md` for the rules that keep these seams contract-tested.
