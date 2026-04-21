package client

import (
	"context"
	stderrors "errors"
	"testing"
	"time"

	"digital.vasic.hypertune/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestSeedReproducibility — identical seed must produce identical BestParams.
func TestSeedReproducibility(t *testing.T) {
	run := func(seed int64) *types.OptimizationResult {
		c, err := New()
		require.NoError(t, err)
		defer c.Close()
		c.SetSeed(seed)
		res, err := c.Optimize(context.Background(), types.ParameterSpace{}, types.OptimizationConfig{
			Model: "m", Prompt: "p", Method: "random", Iterations: 6,
		})
		require.NoError(t, err)
		return res
	}
	a := run(9999)
	b := run(9999)
	assert.Equal(t, a.BestScore, b.BestScore)
	assert.Equal(t, a.BestParams["temperature"], b.BestParams["temperature"])
	assert.Equal(t, a.BestParams["top_p"], b.BestParams["top_p"])
}

// TestSeedDivergenceWithDifferentSeeds — different seeds must usually diverge.
func TestSeedDivergenceWithDifferentSeeds(t *testing.T) {
	c, err := New()
	require.NoError(t, err)
	defer c.Close()

	c.SetSeed(1)
	r1, err := c.Optimize(context.Background(), types.ParameterSpace{}, types.OptimizationConfig{
		Model: "m", Prompt: "p", Method: "random", Iterations: 4,
	})
	require.NoError(t, err)

	c.SetSeed(2)
	r2, err := c.Optimize(context.Background(), types.ParameterSpace{}, types.OptimizationConfig{
		Model: "m", Prompt: "p", Method: "random", Iterations: 4,
	})
	require.NoError(t, err)

	// Not identical in all slots with high probability.
	assert.NotEqual(t, r1.BestParams["temperature"], r2.BestParams["temperature"])
}

// TestOptimizeInvalidConfig — missing model/prompt must error.
func TestOptimizeInvalidConfig(t *testing.T) {
	c, err := New()
	require.NoError(t, err)
	defer c.Close()
	_, err = c.Optimize(context.Background(), types.ParameterSpace{}, types.OptimizationConfig{})
	assert.Error(t, err)
}

// TestOptimizeRunnerErrorPropagates — runner error bubbles up wrapped.
func TestOptimizeRunnerErrorPropagates(t *testing.T) {
	c, err := New()
	require.NoError(t, err)
	defer c.Close()
	c.SetRunner(func(_ context.Context, _ string, _ map[string]float64) (string, time.Duration, error) {
		return "", 0, stderrors.New("backend down")
	})
	_, err = c.Optimize(context.Background(), types.ParameterSpace{}, types.OptimizationConfig{
		Model: "m", Prompt: "p", Method: "random", Iterations: 2,
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "runner failed")
}

// TestRegisterMetricCollision — registering a name twice overrides.
func TestRegisterMetricCollision(t *testing.T) {
	c, err := New()
	require.NoError(t, err)
	defer c.Close()

	c.RegisterMetric("m1", func(_ context.Context, _, _ string) (float64, error) { return 0.1, nil })
	c.RegisterMetric("m1", func(_ context.Context, _, _ string) (float64, error) { return 0.9, nil })

	tr, err := c.Evaluate(context.Background(), map[string]float64{"top_p": 0.9}, "p", "m")
	require.NoError(t, err)
	// Evaluate uses "default" not custom; assert collision itself replaced.
	require.NotNil(t, tr)

	res, err := c.Optimize(context.Background(), types.ParameterSpace{}, types.OptimizationConfig{
		Model: "m", Prompt: "p", Method: "random", Iterations: 1, Metric: "m1",
	})
	require.NoError(t, err)
	assert.InDelta(t, 0.9, res.BestScore, 1e-9)
}

// TestRegisterMetricIgnoresNilOrEmpty.
func TestRegisterMetricIgnoresNilOrEmpty(t *testing.T) {
	c, err := New()
	require.NoError(t, err)
	defer c.Close()
	c.RegisterMetric("", func(_ context.Context, _, _ string) (float64, error) { return 1, nil })
	c.RegisterMetric("x", nil)

	ms, err := c.GetMetrics(context.Background())
	require.NoError(t, err)
	for _, m := range ms {
		assert.NotEqual(t, "", m.Name)
		assert.NotEqual(t, "x", m.Name)
	}
}

// TestSetRunnerNilIgnored — keeps baseline runner.
func TestSetRunnerNilIgnored(t *testing.T) {
	c, err := New()
	require.NoError(t, err)
	defer c.Close()
	c.SetRunner(nil)
	tr, err := c.Evaluate(context.Background(), map[string]float64{"top_p": 0.9}, "hi", "m")
	require.NoError(t, err)
	assert.NotEmpty(t, tr.Output)
}

// TestSuggestParametersPerturbsAroundBest — with ≥3 history, result should be
// within perturbation radius of the best point.
func TestSuggestParametersPerturbsAroundBest(t *testing.T) {
	c, err := New()
	require.NoError(t, err)
	defer c.Close()
	c.SetSeed(17)

	history := []types.TrialResult{
		{Params: map[string]float64{"temperature": 0.5, "top_p": 0.9}, Score: 0.1},
		{Params: map[string]float64{"temperature": 0.7, "top_p": 0.95}, Score: 0.9},
		{Params: map[string]float64{"temperature": 0.3, "top_p": 0.8}, Score: 0.3},
	}
	out, err := c.SuggestParameters(context.Background(), types.ParameterSpace{}, history)
	require.NoError(t, err)
	// Perturbation is ±0.1 around best's 0.7 (within [0.6, 0.8]).
	assert.InDelta(t, 0.7, out["temperature"], 0.2)
	assert.InDelta(t, 0.95, out["top_p"], 0.2)
}

// TestGridSearchProducesNonEmptyGrid — grid dimension is deterministic.
func TestGridSearchProducesNonEmptyGrid(t *testing.T) {
	c, err := New()
	require.NoError(t, err)
	defer c.Close()

	res, err := c.GridSearch(context.Background(), types.ParameterSpace{Temperature: 0.6, TopP: 0.85}, types.OptimizationConfig{
		Model: "m", Prompt: "p",
	})
	require.NoError(t, err)
	// buildGrid: 4 temps × 3 topPs = 12 points
	assert.Equal(t, 12, res.Iterations)
}

// TestBayesianOptimizeReducesToSeedOnShortBudget — 3 or fewer iters should not crash.
func TestBayesianOptimizeReducesToSeedOnShortBudget(t *testing.T) {
	c, err := New()
	require.NoError(t, err)
	defer c.Close()
	res, err := c.BayesianOptimize(context.Background(), types.ParameterSpace{}, types.OptimizationConfig{
		Model: "m", Prompt: "p", Iterations: 2,
	})
	require.NoError(t, err)
	assert.Equal(t, 2, res.Iterations)
}
