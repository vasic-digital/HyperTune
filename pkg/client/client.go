// Package client provides the Go client for the HyperTune library.
//
// HyperTune runs hyperparameter search (random search + grid search + a
// Bayesian-optimisation lite variant) over LLM inference parameters:
// temperature, top_p, top_k, repetition_penalty, presence_penalty,
// frequency_penalty, and max_tokens. An LLM Runner and a Metric are
// injected by the consumer; baseline implementations are provided so
// tests can exercise the search orchestration without a real backend.
//
// Basic usage:
//
//	import hypertune "digital.vasic.hypertune/pkg/client"
//
//	c, err := hypertune.New()
//	if err != nil { log.Fatal(err) }
//	defer c.Close()
package client

import (
	"context"
	"math/rand"
	"strings"
	"sync"
	"time"

	"digital.vasic.pliniuscommon/pkg/config"
	"digital.vasic.pliniuscommon/pkg/errors"

	. "digital.vasic.hypertune/pkg/types"
)

// Runner produces a completion for the given prompt under the given params.
type Runner func(ctx context.Context, prompt string, params map[string]float64) (string, time.Duration, error)

// Metric scores a candidate output. Higher is better unless the metric's
// declared Direction is "minimize".
type Metric func(ctx context.Context, output, reference string) (float64, error)

// Client is the Go client for HyperTune.
type Client struct {
	cfg    *config.Config
	mu     sync.RWMutex
	closed bool

	runner  Runner
	metrics map[string]Metric
	seed    int64
}

// New creates a new HyperTune client with baseline runner + default metrics.
func New(opts ...config.Option) (*Client, error) {
	cfg := config.New("hypertune", opts...)
	if err := cfg.Validate(); err != nil {
		return nil, errors.Wrap(errors.ErrCodeInvalidArgument, "hypertune",
			"invalid configuration", err)
	}
	c := &Client{
		cfg:     cfg,
		runner:  baselineRunner,
		metrics: defaultMetrics(),
		seed:    time.Now().UnixNano(),
	}
	return c, nil
}

// NewFromConfig creates a client from a config object.
func NewFromConfig(cfg *config.Config) (*Client, error) {
	if err := cfg.Validate(); err != nil {
		return nil, errors.Wrap(errors.ErrCodeInvalidArgument, "hypertune",
			"invalid configuration", err)
	}
	return &Client{
		cfg:     cfg,
		runner:  baselineRunner,
		metrics: defaultMetrics(),
		seed:    time.Now().UnixNano(),
	}, nil
}

// Close gracefully closes the client.
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	return nil
}

// Config returns the client configuration.
func (c *Client) Config() *config.Config { return c.cfg }

// SetRunner injects the LLM runner.
func (c *Client) SetRunner(r Runner) {
	if r == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.runner = r
}

// RegisterMetric adds/overrides a metric by name.
func (c *Client) RegisterMetric(name string, m Metric) {
	if name == "" || m == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.metrics[name] = m
}

// SetSeed fixes the random seed used by Optimize/SuggestParameters.
func (c *Client) SetSeed(seed int64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.seed = seed
}

// Optimize dispatches to the appropriate search backend based on cfg.Method.
// Supported methods: "random" (default), "grid", "bayesian".
func (c *Client) Optimize(ctx context.Context, space ParameterSpace, cfg OptimizationConfig) (*OptimizationResult, error) {
	if err := cfg.Validate(); err != nil {
		return nil, errors.Wrap(errors.ErrCodeInvalidArgument, "hypertune",
			"invalid optimization config", err)
	}
	switch strings.ToLower(strings.TrimSpace(cfg.Method)) {
	case "grid":
		return c.GridSearch(ctx, space, cfg)
	case "bayesian", "bayes", "bo":
		return c.BayesianOptimize(ctx, space, cfg)
	case "", "random":
		return c.randomSearch(ctx, space, cfg)
	default:
		return nil, errors.New(errors.ErrCodeInvalidArgument, "hypertune",
			"unknown optimization method: "+cfg.Method)
	}
}

// GridSearch evaluates a cartesian product over a fixed small grid derived
// from the parameter space.
func (c *Client) GridSearch(ctx context.Context, space ParameterSpace, cfg OptimizationConfig) (*OptimizationResult, error) {
	if err := cfg.Validate(); err != nil {
		return nil, errors.Wrap(errors.ErrCodeInvalidArgument, "hypertune",
			"invalid optimization config", err)
	}
	space.Defaults()
	grid := buildGrid(space)
	return c.evaluateAll(ctx, grid, cfg)
}

// BayesianOptimize is a minimal BO-lite: seed with 3 random samples, then
// repeatedly pick the mean-shifted neighbourhood of the current best.
func (c *Client) BayesianOptimize(ctx context.Context, space ParameterSpace, cfg OptimizationConfig) (*OptimizationResult, error) {
	if err := cfg.Validate(); err != nil {
		return nil, errors.Wrap(errors.ErrCodeInvalidArgument, "hypertune",
			"invalid optimization config", err)
	}
	space.Defaults()
	iters := cfg.Iterations
	if iters <= 0 {
		iters = 10
	}
	rng := rand.New(rand.NewSource(c.seed))
	history := make([]map[string]float64, 0, iters)
	// seed with 3 random samples
	for i := 0; i < 3 && i < iters; i++ {
		history = append(history, samplePoint(rng, space))
	}
	// remainder: local perturbation around current best
	for i := len(history); i < iters; i++ {
		// evaluate incrementally to know the current best
		trials, err := c.evaluatePoints(ctx, history, cfg)
		if err != nil {
			return nil, err
		}
		best := bestTrial(trials, cfg)
		history = append(history, perturb(rng, space, best.Params))
	}
	return c.evaluateAll(ctx, history, cfg)
}

// Evaluate runs a single trial.
func (c *Client) Evaluate(ctx context.Context, params map[string]float64, prompt string, model string) (*TrialResult, error) {
	if prompt == "" {
		return nil, errors.New(errors.ErrCodeInvalidArgument, "hypertune", "prompt is required")
	}
	c.mu.RLock()
	runner := c.runner
	c.mu.RUnlock()
	out, latency, err := runner(ctx, prompt, params)
	if err != nil {
		return nil, errors.Wrap(errors.ErrCodeUnavailable, "hypertune",
			"runner failed", err)
	}
	score := c.scoreOutput(ctx, out, "", "")
	return &TrialResult{
		Output:    out,
		Score:     score,
		Params:    copyParams(params),
		LatencyMs: latency.Milliseconds(),
	}, nil
}

// GetMetrics lists registered evaluation metrics.
func (c *Client) GetMetrics(ctx context.Context) ([]EvaluationMetric, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	out := make([]EvaluationMetric, 0, len(c.metrics))
	for name := range c.metrics {
		out = append(out, EvaluationMetric{
			Name:        name,
			Description: "baseline metric: " + name,
			Direction:   "maximize",
		})
	}
	return out, nil
}

// SuggestParameters proposes the next parameter set to try. With fewer than
// 3 history samples it returns a random point; otherwise it returns a
// locally-perturbed version of the current best.
func (c *Client) SuggestParameters(ctx context.Context, space ParameterSpace, history []TrialResult) (map[string]float64, error) {
	space.Defaults()
	rng := rand.New(rand.NewSource(c.seed + int64(len(history))))
	if len(history) < 3 {
		return samplePoint(rng, space), nil
	}
	// find best
	bestIdx := 0
	for i, h := range history {
		if h.Score > history[bestIdx].Score {
			bestIdx = i
		}
	}
	return perturb(rng, space, history[bestIdx].Params), nil
}

// --- internals ---

func (c *Client) randomSearch(ctx context.Context, space ParameterSpace, cfg OptimizationConfig) (*OptimizationResult, error) {
	space.Defaults()
	iters := cfg.Iterations
	if iters <= 0 {
		iters = 12
	}
	rng := rand.New(rand.NewSource(c.seed))
	points := make([]map[string]float64, iters)
	for i := range points {
		points[i] = samplePoint(rng, space)
	}
	return c.evaluateAll(ctx, points, cfg)
}

func (c *Client) evaluateAll(ctx context.Context, points []map[string]float64, cfg OptimizationConfig) (*OptimizationResult, error) {
	start := time.Now()
	trials, err := c.evaluatePoints(ctx, points, cfg)
	if err != nil {
		return nil, err
	}
	best := bestTrial(trials, cfg)
	return &OptimizationResult{
		AllResults: trials,
		BestParams: copyParams(best.Params),
		BestScore:  best.Score,
		Iterations: len(trials),
		TimeMs:     time.Since(start).Milliseconds(),
	}, nil
}

func (c *Client) evaluatePoints(ctx context.Context, points []map[string]float64, cfg OptimizationConfig) ([]TrialResult, error) {
	c.mu.RLock()
	runner := c.runner
	c.mu.RUnlock()

	out := make([]TrialResult, 0, len(points))
	for i, p := range points {
		t0 := time.Now()
		result, latency, err := runner(ctx, cfg.Prompt, p)
		if err != nil {
			return nil, errors.Wrap(errors.ErrCodeUnavailable, "hypertune",
				"runner failed", err)
		}
		if latency == 0 {
			latency = time.Since(t0)
		}
		out = append(out, TrialResult{
			Output:    result,
			Score:     c.scoreOutput(ctx, result, cfg.ReferenceOutput, cfg.Metric),
			Params:    copyParams(p),
			LatencyMs: latency.Milliseconds(),
			Iteration: i,
		})
	}
	return out, nil
}

func (c *Client) scoreOutput(ctx context.Context, output, reference, metricName string) float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if metricName != "" {
		if m, ok := c.metrics[metricName]; ok {
			if s, err := m(ctx, output, reference); err == nil {
				return s
			}
		}
	}
	// default: length + reference overlap
	if m, ok := c.metrics["default"]; ok {
		if s, err := m(ctx, output, reference); err == nil {
			return s
		}
	}
	return 0
}

func bestTrial(trials []TrialResult, cfg OptimizationConfig) TrialResult {
	if len(trials) == 0 {
		return TrialResult{}
	}
	bestIdx := 0
	for i, t := range trials {
		if t.Score > trials[bestIdx].Score {
			bestIdx = i
		}
	}
	return trials[bestIdx]
}

func buildGrid(s ParameterSpace) []map[string]float64 {
	temps := []float64{0.2, 0.5, 0.8, s.Temperature}
	topPs := []float64{0.5, 0.9, s.TopP}
	grid := make([]map[string]float64, 0, len(temps)*len(topPs))
	for _, t := range temps {
		for _, p := range topPs {
			grid = append(grid, map[string]float64{
				"temperature":        t,
				"top_p":              p,
				"top_k":              float64(max0(s.TopK)),
				"max_tokens":         float64(max0(s.MaxTokens)),
				"repetition_penalty": s.RepetitionPenalty,
				"presence_penalty":   s.PresencePenalty,
				"frequency_penalty":  s.FrequencyPenalty,
			})
		}
	}
	return grid
}

func samplePoint(rng *rand.Rand, s ParameterSpace) map[string]float64 {
	return map[string]float64{
		"temperature":        rng.Float64() * 1.5,
		"top_p":              0.5 + rng.Float64()*0.5,
		"top_k":              float64(rng.Intn(50) + 1),
		"max_tokens":         float64(128 + rng.Intn(2048)),
		"repetition_penalty": 1.0 + rng.Float64()*0.5,
		"presence_penalty":   rng.Float64() - 0.5,
		"frequency_penalty":  rng.Float64() - 0.5,
	}
}

func perturb(rng *rand.Rand, s ParameterSpace, base map[string]float64) map[string]float64 {
	out := copyParams(base)
	// perturb each by small Gaussian-like jitter
	keys := []string{"temperature", "top_p", "repetition_penalty",
		"presence_penalty", "frequency_penalty"}
	for _, k := range keys {
		if v, ok := out[k]; ok {
			out[k] = v + (rng.Float64()-0.5)*0.2
		}
	}
	return out
}

func copyParams(p map[string]float64) map[string]float64 {
	out := make(map[string]float64, len(p))
	for k, v := range p {
		out[k] = v
	}
	return out
}

func max0(x int) int {
	if x < 0 {
		return 0
	}
	return x
}

// --- baseline runner / metrics ---

func baselineRunner(_ context.Context, prompt string, params map[string]float64) (string, time.Duration, error) {
	// deterministic: output length is a linear combo of params so tests have a signal
	suffix := ""
	for i := 0; i < int(params["top_p"]*10); i++ {
		suffix += "."
	}
	return prompt + suffix, time.Millisecond, nil
}

func defaultMetrics() map[string]Metric {
	return map[string]Metric{
		"default": func(_ context.Context, output, reference string) (float64, error) {
			// exact match gets 1.0, otherwise prefix-overlap ratio.
			if reference != "" && output == reference {
				return 1.0, nil
			}
			if reference != "" {
				n := 0
				limit := len(output)
				if len(reference) < limit {
					limit = len(reference)
				}
				for i := 0; i < limit; i++ {
					if output[i] == reference[i] {
						n++
					} else {
						break
					}
				}
				if limit == 0 {
					return 0, nil
				}
				return float64(n) / float64(limit), nil
			}
			// no reference: reward moderate length
			L := float64(len(output))
			if L <= 0 {
				return 0, nil
			}
			if L > 500 {
				return 500.0 / L, nil
			}
			return L / 500.0, nil
		},
		"length": func(_ context.Context, output, _ string) (float64, error) {
			return float64(len(output)), nil
		},
		"exact_match": func(_ context.Context, output, reference string) (float64, error) {
			if output == reference {
				return 1.0, nil
			}
			return 0, nil
		},
	}
}
