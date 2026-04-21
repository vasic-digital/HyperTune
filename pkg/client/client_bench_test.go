package client

import (
	"context"
	"testing"

	"digital.vasic.hypertune/pkg/types"
)

func BenchmarkOptimizeRandom(b *testing.B) {
	c, err := New()
	if err != nil {
		b.Fatal(err)
	}
	defer c.Close()
	c.SetSeed(42)
	ctx := context.Background()
	cfg := types.OptimizationConfig{Model: "m", Prompt: "p", Method: "random", Iterations: 4}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := c.Optimize(ctx, types.ParameterSpace{}, cfg); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGridSearch(b *testing.B) {
	c, err := New()
	if err != nil {
		b.Fatal(err)
	}
	defer c.Close()
	ctx := context.Background()
	cfg := types.OptimizationConfig{Model: "m", Prompt: "p"}
	space := types.ParameterSpace{}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := c.GridSearch(ctx, space, cfg); err != nil {
			b.Fatal(err)
		}
	}
}
