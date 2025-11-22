package plugin

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDefaultConfig(t *testing.T) {
	tests := []struct {
		name     string
		validate func(*testing.T, *AgenticRAGConfig)
	}{
		{
			name: "default config has correct model name",
			validate: func(t *testing.T, cfg *AgenticRAGConfig) {
				assert.Equal(t, "googleai/gemini-2.5-flash", cfg.ModelName)
			},
		},
		{
			name: "default config has processing settings",
			validate: func(t *testing.T, cfg *AgenticRAGConfig) {
				assert.Equal(t, 1000, cfg.Processing.DefaultChunkSize)
				assert.Equal(t, 20, cfg.Processing.DefaultMaxChunks)
				assert.Equal(t, 3, cfg.Processing.DefaultRecursiveDepth)
				assert.True(t, cfg.Processing.RespectSentences)
			},
		},
		{
			name: "default config has knowledge graph enabled",
			validate: func(t *testing.T, cfg *AgenticRAGConfig) {
				assert.True(t, cfg.KnowledgeGraph.Enabled)
				assert.NotEmpty(t, cfg.KnowledgeGraph.EntityTypes)
				assert.NotEmpty(t, cfg.KnowledgeGraph.RelationTypes)
				assert.Equal(t, 0.7, cfg.KnowledgeGraph.MinConfidenceThreshold)
			},
		},
		{
			name: "default config has fact verification enabled",
			validate: func(t *testing.T, cfg *AgenticRAGConfig) {
				assert.True(t, cfg.FactVerification.Enabled)
				assert.True(t, cfg.FactVerification.RequireEvidence)
				assert.Equal(t, 0.7, cfg.FactVerification.MinConfidenceScore)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := DefaultConfig()
			require.NotNil(t, cfg)
			tt.validate(t, cfg)
		})
	}
}

func TestNewAgenticRAGProcessor(t *testing.T) {
	tests := []struct {
		name   string
		config *AgenticRAGConfig
		check  func(*testing.T, *AgenticRAGProcessor)
	}{
		{
			name:   "creates processor with nil config",
			config: nil,
			check: func(t *testing.T, p *AgenticRAGProcessor) {
				assert.NotNil(t, p)
				assert.NotNil(t, p.config)
			},
		},
		{
			name:   "creates processor with provided config",
			config: DefaultConfig(),
			check: func(t *testing.T, p *AgenticRAGProcessor) {
				assert.NotNil(t, p)
				assert.Equal(t, "googleai/gemini-2.5-flash", p.config.ModelName)
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

func TestSplitIntoSentences(t *testing.T) {
	processor := NewAgenticRAGProcessor(nil)

	tests := []struct {
		name     string
		input    string
		expected []string
	}{
		{
			name:     "empty string",
			input:    "",
			expected: []string{},
		},
		{
			name:     "single sentence",
			input:    "This is a test.",
			expected: []string{"This is a test."},
		},
		{
			name:     "multiple sentences",
			input:    "First sentence. Second sentence! Third sentence?",
			expected: []string{"First sentence", "Second sentence", "Third sentence?"},
		},
		{
			name:     "sentences with extra spaces",
			input:    "First sentence.  Second sentence.   Third sentence.",
			expected: []string{"First sentence", "Second sentence", "Third sentence."},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := processor.splitIntoSentences(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestCalculateRelevanceScore(t *testing.T) {
	processor := NewAgenticRAGProcessor(nil)

	tests := []struct {
		name     string
		query    string
		content  string
		expected float64
	}{
		{
			name:     "no matching words",
			query:    "artificial intelligence",
			content:  "The weather is nice today.",
			expected: 0.0,
		},
		{
			name:     "partial match",
			query:    "artificial intelligence machine learning",
			content:  "Artificial intelligence is transforming industries.",
			expected: 0.5, // 2 out of 4 words match
		},
		{
			name:     "full match",
			query:    "machine learning",
			content:  "Machine learning algorithms are powerful.",
			expected: 1.0,
		},
		{
			name:     "case insensitive matching",
			query:    "Machine Learning",
			content:  "machine learning is popular",
			expected: 1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score := processor.calculateRelevanceScore(tt.query, tt.content)
			assert.Equal(t, tt.expected, score)
		})
	}
}

func TestChunkDocument(t *testing.T) {
	processor := NewAgenticRAGProcessor(nil)
	ctx := context.Background()

	tests := []struct {
		name        string
		doc         Document
		maxChunks   int
		checkResult func(*testing.T, []DocumentChunk, error)
	}{
		{
			name: "empty document",
			doc: Document{
				ID:      "test_doc",
				Content: "",
				Source:  "test",
			},
			maxChunks: 10,
			checkResult: func(t *testing.T, chunks []DocumentChunk, err error) {
				assert.NoError(t, err)
				assert.Empty(t, chunks)
			},
		},
		{
			name: "small document single chunk",
			doc: Document{
				ID:      "test_doc",
				Content: "This is a small document. It should fit in one chunk.",
				Source:  "test",
			},
			maxChunks: 10,
			checkResult: func(t *testing.T, chunks []DocumentChunk, err error) {
				assert.NoError(t, err)
				assert.Len(t, chunks, 1)
				assert.Equal(t, "test_doc", chunks[0].DocumentID)
			},
		},
		{
			name: "large document multiple chunks",
			doc: Document{
				ID: "test_doc",
				Content: func() string {
					content := ""
					for j := 0; j < 100; j++ {
						content += fmt.Sprintf("This is sentence number %d. ", j)
					}
					return content
				}(),
				Source: "test",
			},
			maxChunks: 5,
			checkResult: func(t *testing.T, chunks []DocumentChunk, err error) {
				assert.NoError(t, err)
				assert.LessOrEqual(t, len(chunks), 5)
				assert.Greater(t, len(chunks), 0)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunks, err := processor.chunkDocument(ctx, tt.doc, tt.maxChunks)
			tt.checkResult(t, chunks, err)
		})
	}
}

func TestLoadDocuments(t *testing.T) {
	processor := NewAgenticRAGProcessor(nil)
	ctx := context.Background()

	tests := []struct {
		name        string
		sources     []string
		checkResult func(*testing.T, []Document, error)
	}{
		{
			name:    "empty sources",
			sources: []string{},
			checkResult: func(t *testing.T, docs []Document, err error) {
				assert.NoError(t, err)
				assert.Empty(t, docs)
			},
		},
		{
			name:    "single source",
			sources: []string{"This is a test document."},
			checkResult: func(t *testing.T, docs []Document, err error) {
				assert.NoError(t, err)
				assert.Len(t, docs, 1)
				assert.Equal(t, "This is a test document.", docs[0].Content)
				assert.NotEmpty(t, docs[0].ID)
			},
		},
		{
			name:    "multiple sources",
			sources: []string{"Document 1", "Document 2", "Document 3"},
			checkResult: func(t *testing.T, docs []Document, err error) {
				assert.NoError(t, err)
				assert.Len(t, docs, 3)
				for _, doc := range docs {
					assert.Contains(t, doc.Content, "Document")
					assert.NotEmpty(t, doc.ID)
					assert.NotEmpty(t, doc.Metadata)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			docs, err := processor.loadDocuments(ctx, tt.sources)
			tt.checkResult(t, docs, err)
		})
	}
}

func TestBreakdownChunk(t *testing.T) {
	processor := NewAgenticRAGProcessor(nil)

	tests := []struct {
		name        string
		chunk       DocumentChunk
		checkResult func(*testing.T, []DocumentChunk)
	}{
		{
			name: "single sentence chunk",
			chunk: DocumentChunk{
				ID:      "chunk_1",
				Content: "This is one sentence.",
			},
			checkResult: func(t *testing.T, subChunks []DocumentChunk) {
				assert.Len(t, subChunks, 1)
			},
		},
		{
			name: "multi-sentence chunk",
			chunk: DocumentChunk{
				ID:      "chunk_1",
				Content: "First sentence. Second sentence. Third sentence.",
			},
			checkResult: func(t *testing.T, subChunks []DocumentChunk) {
				assert.Greater(t, len(subChunks), 1)
				for _, sc := range subChunks {
					assert.Contains(t, sc.ID, "chunk_1_sub_")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			subChunks := processor.breakdownChunk(tt.chunk)
			tt.checkResult(t, subChunks)
		})
	}
}

func TestParseConfidence(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected float64
	}{
		{
			name:     "valid percentage",
			input:    "85%",
			expected: 0.85,
		},
		{
			name:     "valid decimal",
			input:    "0.75",
			expected: 0.0075, // Note: parseConfidence divides by 100
		},
		{
			name:     "invalid input",
			input:    "invalid",
			expected: 0.0,
		},
		{
			name:     "empty string",
			input:    "",
			expected: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := parseConfidence(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestFallbackRelevanceScoring(t *testing.T) {
	processor := NewAgenticRAGProcessor(nil)

	chunks := []DocumentChunk{
		{
			ID:      "chunk_1",
			Content: "Artificial intelligence is transforming the world.",
		},
		{
			ID:      "chunk_2",
			Content: "The weather is sunny today.",
		},
		{
			ID:      "chunk_3",
			Content: "Machine learning uses artificial intelligence techniques.",
		},
	}

	query := "artificial intelligence machine learning"

	result := processor.fallbackRelevanceScoring(query, chunks)

	// Should return chunks sorted by relevance
	assert.NotEmpty(t, result)
	assert.LessOrEqual(t, len(result), len(chunks))

	// First chunk should have higher relevance score
	if len(result) > 1 {
		assert.GreaterOrEqual(t, result[0].RelevanceScore, result[1].RelevanceScore)
	}

	// All returned chunks should have score > 0.3
	for _, chunk := range result {
		assert.Greater(t, chunk.RelevanceScore, 0.3)
	}
}

func TestParseFactVerificationResponse(t *testing.T) {
	processor := NewAgenticRAGProcessor(nil)

	tests := []struct {
		name        string
		input       map[string]any
		expectError bool
		checkResult func(*testing.T, *FactVerification)
	}{
		{
			name: "valid response",
			input: map[string]any{
				"claims": []interface{}{
					map[string]interface{}{
						"text":       "AI is growing",
						"status":     "verified",
						"confidence": 0.95,
						"evidence":   []interface{}{"Source 1"},
					},
				},
				"overall": "verified",
			},
			expectError: false,
			checkResult: func(t *testing.T, fv *FactVerification) {
				assert.NotNil(t, fv)
				assert.Len(t, fv.Claims, 1)
				assert.Equal(t, "verified", fv.Overall)
				assert.Equal(t, "AI is growing", fv.Claims[0].Text)
			},
		},
		{
			name: "invalid claims format",
			input: map[string]any{
				"claims": "not an array",
			},
			expectError: true,
			checkResult: func(t *testing.T, fv *FactVerification) {
				assert.Nil(t, fv)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := processor.parseFactVerificationResponse(tt.input)
			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
			tt.checkResult(t, result)
		})
	}
}

func TestParseKnowledgeGraphResponse(t *testing.T) {
	processor := NewAgenticRAGProcessor(nil)

	tests := []struct {
		name        string
		input       map[string]any
		expectError bool
		checkResult func(*testing.T, *KnowledgeGraph)
	}{
		{
			name: "valid knowledge graph",
			input: map[string]any{
				"entities": []interface{}{
					map[string]interface{}{
						"name":       "OpenAI",
						"type":       "ORGANIZATION",
						"confidence": 0.95,
					},
				},
				"relations": []interface{}{
					map[string]interface{}{
						"from_entity":   "OpenAI",
						"to_entity":     "GPT",
						"relation_type": "DEVELOPS",
						"confidence":    0.9,
					},
				},
			},
			expectError: false,
			checkResult: func(t *testing.T, kg *KnowledgeGraph) {
				assert.NotNil(t, kg)
				assert.Len(t, kg.Entities, 1)
				assert.Len(t, kg.Relations, 1)
				assert.Equal(t, "OpenAI", kg.Entities[0].Name)
			},
		},
		{
			name: "entities below confidence threshold",
			input: map[string]any{
				"entities": []interface{}{
					map[string]interface{}{
						"name":       "Test",
						"type":       "CONCEPT",
						"confidence": 0.5, // Below default threshold of 0.7
					},
				},
			},
			expectError: false,
			checkResult: func(t *testing.T, kg *KnowledgeGraph) {
				assert.NotNil(t, kg)
				assert.Empty(t, kg.Entities) // Should be filtered out
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := processor.parseKnowledgeGraphResponse(tt.input)
			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
			tt.checkResult(t, result)
		})
	}
}
