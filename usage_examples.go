package main

import (
	"fmt"
	"log"
)

// RunUsageExamples demonstrates different ways to use the transformer
func RunUsageExamples() {
	fmt.Println("🚀 Go Transformer Usage Examples")
	fmt.Println("=================================")

	// Run different usage examples
	ExampleClassification()
	ExampleTextCompletion()
	ExampleBatchProcessing()
	ExampleCustomConfigurations()
	ExampleWithTokenizer()

	fmt.Println("\n🎉 All usage examples completed!")
	fmt.Println("\n💡 Tips for using the transformer:")
	fmt.Println("   1. Adjust VocabSize based on your tokenizer")
	fmt.Println("   2. Use smaller models (fewer layers/dimensions) for faster experimentation")
	fmt.Println("   3. Increase SeqLen for longer sequences")
	fmt.Println("   4. Temperature in Generate() controls randomness (0.1=focused, 1.0=creative)")
	fmt.Println("   5. The current implementation is simplified - add training for real applications")
}

// Example 1: Text Classification/Sentiment Analysis
func ExampleClassification() {
	fmt.Println("=== Text Classification Example ===")

	config := TransformerConfig{
		VocabSize:    1000,
		SeqLen:       64,  // Shorter sequences for classification
		DModel:       256, // Smaller model for classification
		NumHeads:     4,
		NumLayers:    3,
		DFF:          1024,
		DropoutRate:  0.1,
		LearningRate: 0.001,
	}

	model, err := NewTransformerModel(config)
	if err != nil {
		log.Fatal(err)
	}
	defer model.Cleanup()

	// Example: classify sentiment of "good movie"
	// In practice, you'd tokenize real text first
	tokens := []int{245, 678} // "good" = 245, "movie" = 678

	output, err := model.Forward(tokens)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Classification logits shape: %dx%d\n", output.Rows(), output.Cols())
	// The last token's output could be used for classification
	fmt.Println("✅ Classification complete")
}

// Example 2: Text Completion
func ExampleTextCompletion() {
	fmt.Println("\n=== Text Completion Example ===")

	config := TransformerConfig{
		VocabSize:    1000,
		SeqLen:       128,
		DModel:       512,
		NumHeads:     8,
		NumLayers:    6,
		DFF:          2048,
		DropoutRate:  0.1,
		LearningRate: 0.0001,
	}

	model, err := NewTransformerModel(config)
	if err != nil {
		log.Fatal(err)
	}
	defer model.Cleanup()

	// Start with a prompt and complete the text
	prompt := []int{10, 25, 47} // Some starting tokens

	fmt.Printf("Starting prompt: %v\n", prompt)

	// Generate completion
	completed, err := model.Generate(prompt, 20, 0.8) // temperature = 0.8 for creativity
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Completed text tokens: %v\n", completed)
	fmt.Printf("Generated %d new tokens\n", len(completed)-len(prompt))
	fmt.Println("✅ Text completion complete")
}

// Example 3: Batch Processing
func ExampleBatchProcessing() {
	fmt.Println("\n=== Batch Processing Example ===")

	config := TransformerConfig{
		VocabSize:    1000,
		SeqLen:       32,
		DModel:       256,
		NumHeads:     4,
		NumLayers:    2,
		DFF:          512,
		DropoutRate:  0.1,
		LearningRate: 0.001,
	}

	model, err := NewTransformerModel(config)
	if err != nil {
		log.Fatal(err)
	}
	defer model.Cleanup()

	// Process multiple sequences
	sequences := [][]int{
		{1, 2, 3, 4},
		{5, 6, 7, 8, 9},
		{10, 11, 12},
	}

	fmt.Printf("Processing %d sequences:\n", len(sequences))

	for i, seq := range sequences {
		output, err := model.Forward(seq)
		if err != nil {
			log.Printf("Error processing sequence %d: %v", i, err)
			continue
		}
		fmt.Printf("  Sequence %d: %v -> output shape: %dx%d\n",
			i+1, seq, output.Rows(), output.Cols())
	}
	fmt.Println("✅ Batch processing complete")
}

// Example 4: Custom Configuration for Different Tasks
func ExampleCustomConfigurations() {
	fmt.Println("\n=== Custom Configuration Examples ===")

	// Small model for quick experimentation
	smallConfig := TransformerConfig{
		VocabSize:    500,
		SeqLen:       32,
		DModel:       128,
		NumHeads:     2,
		NumLayers:    2,
		DFF:          256,
		DropoutRate:  0.1,
		LearningRate: 0.01,
	}

	// Large model for serious applications
	largeConfig := TransformerConfig{
		VocabSize:    10000,
		SeqLen:       512,
		DModel:       768,
		NumHeads:     12,
		NumLayers:    12,
		DFF:          3072,
		DropoutRate:  0.1,
		LearningRate: 0.0001,
	}

	fmt.Printf("Small model parameters: %d\n", calculateModelParameters(smallConfig))
	fmt.Printf("Large model parameters: %d\n", calculateModelParameters(largeConfig))

	// You can create models with different configurations for different tasks
	fmt.Println("✅ Configuration examples complete")
}

// Example 5: Integration with Real Text Processing
func ExampleWithTokenizer() {
	fmt.Println("\n=== Integration with Tokenizer Example ===")

	// Create tokenizer
	tokenizer := NewTokenizer()

	// Build vocabulary from sample texts
	sampleTexts := []string{
		"hello world",
		"good morning",
		"transformer model",
		"artificial intelligence",
		"machine learning is great",
		"natural language processing",
	}

	tokenizer.BuildVocabulary(sampleTexts, 1000)

	fmt.Printf("Vocabulary size: %d\n", tokenizer.VocabSize)

	// Create model with tokenizer's vocabulary size
	config := TransformerConfig{
		VocabSize:    tokenizer.VocabSize,
		SeqLen:       64,
		DModel:       256,
		NumHeads:     4,
		NumLayers:    3,
		DFF:          512,
		DropoutRate:  0.1,
		LearningRate: 0.001,
	}

	model, err := NewTransformerModel(config)
	if err != nil {
		log.Fatal(err)
	}
	defer model.Cleanup()

	// Process real text
	testText := "hello world"
	tokens := tokenizer.Encode(testText)
	fmt.Printf("Text: '%s' -> Tokens: %v\n", testText, tokens)

	if len(tokens) > 0 {
		output, err := model.Forward(tokens)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Processed text, output shape: %dx%d\n", output.Rows(), output.Cols())

		// Decode the tokens back to verify
		decoded := tokenizer.Decode(tokens)
		fmt.Printf("Decoded back: '%s'\n", decoded)
	}

	fmt.Println("✅ Tokenizer integration complete")
}

func runExamples() {
	RunUsageExamples()
}
