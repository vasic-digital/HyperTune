package client

import (
	"context"
	"testing"
	"time"

	"digital.vasic.hypertune/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNew(t *testing.T) {
	client, err := New()
	require.NoError(t, err)
	assert.NotNil(t, client)
	assert.NoError(t, client.Close())
}

func TestDoubleClose(t *testing.T) {
	client, err := New()
	require.NoError(t, err)
	assert.NoError(t, client.Close())
	assert.NoError(t, client.Close())
}

func TestConfig(t *testing.T) {
	client, err := New()
	require.NoError(t, err)
	defer client.Close()
	assert.NotNil(t, client.Config())
}

func TestOptimizeRandom(t *testing.T) {
	c, err := New()
	require.NoError(t, err)
	defer c.Close()
	c.SetSeed(42)

	res, err := c.Optimize(context.Background(), types.ParameterSpace{}, types.OptimizationConfig{
		Model: "gpt-4", Prompt: "hello", Method: "random", Iterations: 5,
	})
	require.NoError(t, err)
	assert.Equal(t, 5, res.Iterations)
	assert.NotNil(t, res.BestParams)
}

func TestOptimizeGrid(t *testing.T) {
	c, err := New()
	require.NoError(t, err)
	defer c.Close()

	res, err := c.Optimize(context.Background(), types.ParameterSpace{}, types.OptimizationConfig{
		Model: "gpt-4", Prompt: "hello", Method: "grid",
	})
	require.NoError(t, err)
	assert.Greater(t, res.Iterations, 0)
	assert.Contains(t, res.BestParams, "temperature")
}

func TestOptimizeBayesian(t *testing.T) {
	c, err := New()
	require.NoError(t, err)
	defer c.Close()
	c.SetSeed(7)

	res, err := c.Optimize(context.Background(), types.ParameterSpace{}, types.OptimizationConfig{
		Model: "gpt-4", Prompt: "hello", Method: "bayesian", Iterations: 6,
	})
	require.NoError(t, err)
	assert.Equal(t, 6, res.Iterations)
}

func TestOptimizeUnknownMethod(t *testing.T) {
	c, err := New()
	require.NoError(t, err)
	defer c.Close()

	_, err = c.Optimize(context.Background(), types.ParameterSpace{}, types.OptimizationConfig{
		Model: "gpt-4", Prompt: "hello", Method: "gradient-descent",
	})
	assert.Error(t, err)
}

func TestEvaluate(t *testing.T) {
	c, err := New()
	require.NoError(t, err)
	defer c.Close()

	tr, err := c.Evaluate(context.Background(), map[string]float64{"top_p": 0.9}, "hello", "gpt-4")
	require.NoError(t, err)
	assert.NotEmpty(t, tr.Output)
	assert.GreaterOrEqual(t, tr.LatencyMs, int64(0))
}

func TestEvaluateEmptyPrompt(t *testing.T) {
	c, err := New()
	require.NoError(t, err)
	defer c.Close()
	_, err = c.Evaluate(context.Background(), map[string]float64{}, "", "gpt-4")
	assert.Error(t, err)
}

func TestGetMetrics(t *testing.T) {
	c, err := New()
	require.NoError(t, err)
	defer c.Close()

	ms, err := c.GetMetrics(context.Background())
	require.NoError(t, err)
	assert.NotEmpty(t, ms)
}

func TestSuggestParameters(t *testing.T) {
	c, err := New()
	require.NoError(t, err)
	defer c.Close()
	c.SetSeed(1)

	p, err := c.SuggestParameters(context.Background(), types.ParameterSpace{}, nil)
	require.NoError(t, err)
	assert.Contains(t, p, "temperature")

	history := []types.TrialResult{
		{Params: map[string]float64{"temperature": 0.5, "top_p": 0.9}, Score: 0.1},
		{Params: map[string]float64{"temperature": 0.7, "top_p": 0.95}, Score: 0.5},
		{Params: map[string]float64{"temperature": 0.3, "top_p": 0.8}, Score: 0.3},
	}
	p2, err := c.SuggestParameters(context.Background(), types.ParameterSpace{}, history)
	require.NoError(t, err)
	assert.Contains(t, p2, "temperature")
}

func TestSetRunnerAndRegisterMetric(t *testing.T) {
	c, err := New()
	require.NoError(t, err)
	defer c.Close()
	c.SetSeed(123)

	c.SetRunner(func(_ context.Context, prompt string, _ map[string]float64) (string, time.Duration, error) {
		return prompt + "-custom", time.Millisecond, nil
	})
	c.RegisterMetric("always_one", func(_ context.Context, _, _ string) (float64, error) {
		return 1.0, nil
	})

	res, err := c.Optimize(context.Background(), types.ParameterSpace{}, types.OptimizationConfig{
		Model: "x", Prompt: "hi", Method: "random", Iterations: 3, Metric: "always_one",
	})
	require.NoError(t, err)
	assert.InDelta(t, 1.0, res.BestScore, 1e-9)
}
