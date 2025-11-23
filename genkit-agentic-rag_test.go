package genkit_agentic_rag

import (
	"testing"

	"github.com/ZanzyTHEbar/genkit-agentic-rag/plugin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaultAgenticRAGConfig(t *testing.T) {
	config := DefaultAgenticRAGConfig()
	require.NotNil(t, config)
	assert.Equal(t, "googleai/gemini-2.5-flash", config.ModelName)
	assert.True(t, config.Processing.RespectSentences)
	assert.True(t, config.KnowledgeGraph.Enabled)
	assert.True(t, config.FactVerification.Enabled)
}

func TestNewAgenticRAGProcessor(t *testing.T) {
	tests := []struct {
		name   string
		config *plugin.AgenticRAGConfig
		check  func(*testing.T, *plugin.AgenticRAGProcessor)
	}{
		{
			name:   "creates processor with nil config",
			config: nil,
			check: func(t *testing.T, p *plugin.AgenticRAGProcessor) {
				assert.NotNil(t, p)
			},
		},
		{
			name:   "creates processor with default config",
			config: DefaultAgenticRAGConfig(),
			check: func(t *testing.T, p *plugin.AgenticRAGProcessor) {
				assert.NotNil(t, p)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			processor := NewAgenticRAGProcessor(tt.config)
			tt.check(t, processor)
		})
	}
}
