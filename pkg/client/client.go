// Package client provides the Go client for the HyperTune library.
// Go library for HyperTune providing automated hyperparameter optimization for LLM inference including temperature, top_p, top_k, repetition penalty, and context window tuning via Bayesian optimization and grid search.
//
// Basic usage:
//
//	import hypertune "digital.vasic.hypertune/pkg/client"
//
//	client, err := hypertune.New()
//	if err != nil { log.Fatal(err) }
//	defer client.Close()
package client

import (
	"context"

	"digital.vasic.pliniuscommon/pkg/config"
	"digital.vasic.pliniuscommon/pkg/errors"
	. "digital.vasic.hypertune/pkg/types"
)

// Client is the Go client for the HyperTune service.
type Client struct {
	cfg    *config.Config
	closed bool
}

// New creates a new HyperTune client.
func New(opts ...config.Option) (*Client, error) {
	cfg := config.New("hypertune", opts...)
	if err := cfg.Validate(); err != nil {
		return nil, errors.Wrap(errors.ErrCodeInvalidArgument, "hypertune",
			"invalid configuration", err)
	}
	return &Client{cfg: cfg}, nil
}

// NewFromConfig creates a client from a config object.
func NewFromConfig(cfg *config.Config) (*Client, error) {
	if err := cfg.Validate(); err != nil {
		return nil, errors.Wrap(errors.ErrCodeInvalidArgument, "hypertune",
			"invalid configuration", err)
	}
	return &Client{cfg: cfg}, nil
}

// Close gracefully closes the client.
func (c *Client) Close() error {
	if c.closed { return nil }
	c.closed = true
	return nil
}

// Config returns the client configuration.
func (c *Client) Config() *config.Config { return c.cfg }

// Optimize Run hyperparameter optimization.
func (c *Client) Optimize(ctx context.Context, space ParameterSpace, cfg OptimizationConfig) (*OptimizationResult, error) {
	return nil, errors.New(errors.ErrCodeUnimplemented, "hypertune",
		"Optimize requires backend service integration")
}

// GridSearch Run grid search.
func (c *Client) GridSearch(ctx context.Context, space ParameterSpace, cfg OptimizationConfig) (*OptimizationResult, error) {
	return nil, errors.New(errors.ErrCodeUnimplemented, "hypertune",
		"GridSearch requires backend service integration")
}

// BayesianOptimize Run Bayesian optimization.
func (c *Client) BayesianOptimize(ctx context.Context, space ParameterSpace, cfg OptimizationConfig) (*OptimizationResult, error) {
	return nil, errors.New(errors.ErrCodeUnimplemented, "hypertune",
		"BayesianOptimize requires backend service integration")
}

// Evaluate Evaluate parameter set.
func (c *Client) Evaluate(ctx context.Context, params map[string]float64, prompt string, model string) (*TrialResult, error) {
	return nil, errors.New(errors.ErrCodeUnimplemented, "hypertune",
		"Evaluate requires backend service integration")
}

// GetMetrics List available metrics.
func (c *Client) GetMetrics(ctx context.Context) ([]EvaluationMetric, error) {
	return nil, errors.New(errors.ErrCodeUnimplemented, "hypertune",
		"GetMetrics requires backend service integration")
}

// SuggestParameters Suggest next parameters to try.
func (c *Client) SuggestParameters(ctx context.Context, space ParameterSpace, history []TrialResult) (map[string]float64, error) {
	return nil, errors.New(errors.ErrCodeUnimplemented, "hypertune",
		"SuggestParameters requires backend service integration")
}

