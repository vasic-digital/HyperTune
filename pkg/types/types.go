// Package types defines Go types for the HyperTune library.
// Go library for HyperTune providing automated hyperparameter optimization for LLM inference including temperature, top_p, top_k, repetition penalty, and context window tuning via Bayesian optimization and grid search.
package types

import (
	"fmt"
	"strings"
)

// ParameterSpace represents parameterspace data.
type ParameterSpace struct {
	FrequencyPenalty float64
	MaxTokens int
	TopP float64
	PresencePenalty float64
	RepetitionPenalty float64
	TopK int
	Temperature float64
}

// Defaults applies default values for unset fields.
func (o *ParameterSpace) Defaults() {
	if o.MaxTokens == 0 { o.MaxTokens = 2048 }
	if o.TopP == 0 { o.TopP = 1.0 }
	if o.Temperature == 0 { o.Temperature = 0.7 }
}

// OptimizationConfig represents optimizationconfig data.
type OptimizationConfig struct {
	Model string
	Method string
	Metric string
	Prompt string
	Iterations int
	ReferenceOutput string
}

// Validate checks that the OptimizationConfig is valid.
func (o *OptimizationConfig) Validate() error {
	if strings.TrimSpace(o.Model) == "" {
		return fmt.Errorf("model is required")
	}
	if strings.TrimSpace(o.Prompt) == "" {
		return fmt.Errorf("prompt is required")
	}
	return nil
}

// OptimizationResult represents optimizationresult data.
type OptimizationResult struct {
	AllResults []TrialResult
	BestParams map[string]float64
	BestScore float64
	Iterations int
	TimeMs int64
}

// TrialResult represents trialresult data.
type TrialResult struct {
	Output string
	Score float64
	Params map[string]float64
	LatencyMs int64
	Iteration int
}

// EvaluationMetric represents evaluationmetric data.
type EvaluationMetric struct {
	Description string
	Direction string
	Name string
}

// Validate checks that the EvaluationMetric is valid.
func (o *EvaluationMetric) Validate() error {
	if strings.TrimSpace(o.Description) == "" {
		return fmt.Errorf("description is required")
	}
	if strings.TrimSpace(o.Name) == "" {
		return fmt.Errorf("name is required")
	}
	return nil
}

