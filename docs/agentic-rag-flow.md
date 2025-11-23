# Agentic RAG Flow

OpenAI's Agentic RAG Flow is a framework that combines retrieval-augmented generation (RAG) with agentic capabilities. This allows for the generation of responses based on both pre-existing knowledge and real-time information retrieval.

## Key Components

1. **Load the Entire Document** into the context window.
   1. Logic to determine the selected model's context window
   2. Appropriately and intelligently manage context window using advanced prompting techniques
2. **Chunk the Document** into manageable chunks that respect sentence boundaries.
   1. Sentence-aware chunking strategy
   2. Configurable chunk sizes
3. **Prompt the model** for which chunks might contain relevant information.
4. **Drill down** into the selected relevant chunks.
5. **Recursively call this function** until we reach paragraph-level content.
6. **Build Knowledge Graph** based on context
   1. Entity extraction
   2. Relationship mapping
   3. Confidence scoring
7. **Verify the answer** for factual accuracy.
8. **Generate a response** based on the retrieved information & context.

## Implementation Features

- Advanced Prompt Engineering techniques
- Dynamic Prompt Templating and Chaining
- Generate Custom Prompts
- Send Prompt to LLM
- RAG Workflow Integration - Context Augmentation
- LLM Final Response Generation
- Structured Output - JSON Schema Validation
- Observability and Metrics Logging
- Robust Error Handling with errbuilder-go
- Return Final Response to Application

## Architecture

- Hexagonal Architecture
- Functional Options Pattern
- Go Best Practices
- Table-driven tests for reliability

