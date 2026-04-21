package types

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParameterSpaceDefaults(t *testing.T) {
	opts := ParameterSpace{}
	opts.Defaults()
	assert.Equal(t, 2048, opts.MaxTokens)
	assert.Equal(t, 0.7, opts.Temperature)
}

func TestOptimizationConfigValidateValid(t *testing.T) {
	opts := OptimizationConfig{
		Model:           "gpt-4",
		Method:          "test",
		Metric:          "test",
		Prompt:          "test prompt",
		ReferenceOutput: "test",
	}
	assert.NoError(t, opts.Validate())
}

func TestOptimizationConfigValidateEmpty(t *testing.T) {
	opts := OptimizationConfig{}
	err := opts.Validate()
	assert.Error(t, err)
}

func TestEvaluationMetricValidateValid(t *testing.T) {
	opts := EvaluationMetric{
		Description: "test description",
		Direction:   "test",
		Name:        "Test Name",
	}
	assert.NoError(t, opts.Validate())
}

func TestEvaluationMetricValidateEmpty(t *testing.T) {
	opts := EvaluationMetric{}
	err := opts.Validate()
	assert.Error(t, err)
}
