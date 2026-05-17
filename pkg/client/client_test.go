package client

import (
	"context"
	"errors"
	"testing"
	"time"

	"digital.vasic.hypertune/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// dotsTestRunner is a deterministic unit-test stand-in for a real LLM Runner.
// CONST-050(A) permits mocks/stubs in unit tests only — production code MUST
// receive a real LLM-dispatching Runner via SetRunner, otherwise New()'s
// default returns ErrBaselineRunnerNotConfigured (round-23 §11.4 audit fix).
// Output length is a linear combination of top_p so tests have a signal.
func dotsTestRunner(_ context.Context, prompt string, params map[string]float64) (string, time.Duration, error) {
	suffix := ""
	for i := 0; i < int(params["top_p"]*10); i++ {
		suffix += "."
	}
	return prompt + suffix, time.Millisecond, nil
}

// newTestClient builds a client with the dots stub installed so unit tests
// have deterministic behaviour without depending on a real LLM provider.
func newTestClient(t *testing.T) *Client {
	t.Helper()
	c, err := New()
	require.NoError(t, err)
	c.SetRunner(dotsTestRunner)
	return c
}

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
	c := newTestClient(t)
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
	c := newTestClient(t)
	defer c.Close()

	res, err := c.Optimize(context.Background(), types.ParameterSpace{}, types.OptimizationConfig{
		Model: "gpt-4", Prompt: "hello", Method: "grid",
	})
	require.NoError(t, err)
	assert.Greater(t, res.Iterations, 0)
	assert.Contains(t, res.BestParams, "temperature")
}

func TestOptimizeBayesian(t *testing.T) {
	c := newTestClient(t)
	defer c.Close()
	c.SetSeed(7)

	res, err := c.Optimize(context.Background(), types.ParameterSpace{}, types.OptimizationConfig{
		Model: "gpt-4", Prompt: "hello", Method: "bayesian", Iterations: 6,
	})
	require.NoError(t, err)
	assert.Equal(t, 6, res.Iterations)
}

func TestOptimizeUnknownMethod(t *testing.T) {
	c := newTestClient(t)
	defer c.Close()

	_, err := c.Optimize(context.Background(), types.ParameterSpace{}, types.OptimizationConfig{
		Model: "gpt-4", Prompt: "hello", Method: "gradient-descent",
	})
	assert.Error(t, err)
}

func TestEvaluate(t *testing.T) {
	c := newTestClient(t)
	defer c.Close()

	tr, err := c.Evaluate(context.Background(), map[string]float64{"top_p": 0.9}, "hello", "gpt-4")
	require.NoError(t, err)
	assert.NotEmpty(t, tr.Output)
	assert.GreaterOrEqual(t, tr.LatencyMs, int64(0))
}

func TestEvaluateEmptyPrompt(t *testing.T) {
	c := newTestClient(t)
	defer c.Close()
	_, err := c.Evaluate(context.Background(), map[string]float64{}, "", "gpt-4")
	assert.Error(t, err)
}

func TestGetMetrics(t *testing.T) {
	c := newTestClient(t)
	defer c.Close()

	ms, err := c.GetMetrics(context.Background())
	require.NoError(t, err)
	assert.NotEmpty(t, ms)
}

func TestSuggestParameters(t *testing.T) {
	c := newTestClient(t)
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

// TestOptimizeWithoutInjectedRunner_ReturnsSentinel asserts the round-23 §11.4
// audit fix: New()'s default Runner returns ErrBaselineRunnerNotConfigured
// when SetRunner is not called, instead of the previous silent dot-padding
// echo that produced fabricated optimisation data.
func TestOptimizeWithoutInjectedRunner_ReturnsSentinel(t *testing.T) {
	c, err := New()
	require.NoError(t, err)
	defer c.Close()

	_, err = c.Optimize(context.Background(), types.ParameterSpace{}, types.OptimizationConfig{
		Model: "gpt-4", Prompt: "hello", Method: "random", Iterations: 3,
	})
	require.Error(t, err, "Optimize without injected Runner MUST surface the sentinel error, not return fabricated data")
	require.True(t, errors.Is(err, ErrBaselineRunnerNotConfigured), "wrapped error MUST be ErrBaselineRunnerNotConfigured; got %v", err)
}

// TestGridSearchWithoutInjectedRunner_ReturnsSentinel — sentinel propagates
// through the grid-search path.
func TestGridSearchWithoutInjectedRunner_ReturnsSentinel(t *testing.T) {
	c, err := New()
	require.NoError(t, err)
	defer c.Close()

	_, err = c.GridSearch(context.Background(), types.ParameterSpace{}, types.OptimizationConfig{
		Model: "gpt-4", Prompt: "hello",
	})
	require.Error(t, err)
	require.True(t, errors.Is(err, ErrBaselineRunnerNotConfigured))
}

// TestBayesianOptimizeWithoutInjectedRunner_ReturnsSentinel — sentinel
// propagates through the BO-lite path.
func TestBayesianOptimizeWithoutInjectedRunner_ReturnsSentinel(t *testing.T) {
	c, err := New()
	require.NoError(t, err)
	defer c.Close()

	_, err = c.BayesianOptimize(context.Background(), types.ParameterSpace{}, types.OptimizationConfig{
		Model: "gpt-4", Prompt: "hello", Iterations: 4,
	})
	require.Error(t, err)
	require.True(t, errors.Is(err, ErrBaselineRunnerNotConfigured))
}

// TestEvaluateWithoutInjectedRunner_ReturnsSentinel — sentinel propagates
// through the single-trial Evaluate path.
func TestEvaluateWithoutInjectedRunner_ReturnsSentinel(t *testing.T) {
	c, err := New()
	require.NoError(t, err)
	defer c.Close()

	_, err = c.Evaluate(context.Background(), map[string]float64{"top_p": 0.9}, "hi", "m")
	require.Error(t, err)
	require.True(t, errors.Is(err, ErrBaselineRunnerNotConfigured))
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
