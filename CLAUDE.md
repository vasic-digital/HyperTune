# CLAUDE.md -- digital.vasic.hypertune

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
