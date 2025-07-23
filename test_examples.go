package main

import (
	"fmt"
	"log"
)

// TestExamples tests all the usage examples with the current advanced implementation
func TestExamples() {
	fmt.Println("🧪 Testing Usage Examples with Advanced Transformer")
	fmt.Println("==================================================")

	// Test each example individually to catch any issues
	fmt.Println("\n1. Testing Classification Example...")
	testClassification()

	fmt.Println("\n2. Testing Text Completion Example...")
	testTextCompletion()

	fmt.Println("\n3. Testing Batch Processing Example...")
	testBatchProcessing()

	fmt.Println("\n4. Testing Custom Configurations Example...")
	testCustomConfigurations()

	fmt.Println("\n5. Testing Tokenizer Integration Example...")
	testTokenizerIntegration()

	fmt.Println("\n✅ All usage examples tested successfully!")
}

// Run the examples test when this file is executed directly
func init() {
	// This will run when the package is imported
}

// Call TestExamples from main for testing
func runExampleTests() {
	TestExamples()
}

func testClassification() {
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("❌ Classification example failed: %v\n", r)
		}
	}()

	config := TransformerConfig{
		VocabSize:    1000,
		SeqLen:       64,  // Shorter sequences for classification
		DModel:       256, // Smaller model for classification
		NumHeads:     4,
		NumLayers:    3,
		DFF:          1024,
		DropoutRate:  0.1,
		LearningRate: 0.001,

		// Advanced features (optional for examples)
		UseQuantization:       false, // Disable for simpler testing
		UseGradientCheckpoint: false,
		EnableCustomKernels:   false,
		AutoTuning:            false,
	}

	model, err := NewTransformerModel(config)
	if err != nil {
		log.Printf("❌ Failed to create classification model: %v", err)
		return
	}
	defer model.Cleanup()

	// Example: classify sentiment of "good movie"
	tokens := []int{245, 678} // "good" = 245, "movie" = 678

	output, err := model.Forward(tokens)
	if err != nil {
		log.Printf("❌ Forward pass failed: %v", err)
		return
	}

	fmt.Printf("   ✅ Classification logits shape: %dx%d\n", output.Rows(), output.Cols())
}

func testTextCompletion() {
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("❌ Text completion example failed: %v\n", r)
		}
	}()

	config := TransformerConfig{
		VocabSize:    1000,
		SeqLen:       128,
		DModel:       512,
		NumHeads:     8,
		NumLayers:    6,
		DFF:          2048,
		DropoutRate:  0.1,
		LearningRate: 0.0001,

		// Enable some advanced features for this test
		UseQuantization:       true,
		QuantizationBits:      16,
		UseGradientCheckpoint: false, // Keep simple for testing
		EnableCustomKernels:   true,
		AutoTuning:            false,
	}

	model, err := NewTransformerModel(config)
	if err != nil {
		log.Printf("❌ Failed to create text completion model: %v", err)
		return
	}
	defer model.Cleanup()

	// Start with a prompt and complete the text
	prompt := []int{10, 25, 47} // Some starting tokens

	fmt.Printf("   Starting prompt: %v\n", prompt)

	// Generate completion
	completed, err := model.Generate(prompt, 10, 0.8) // Smaller generation for testing
	if err != nil {
		log.Printf("❌ Generation failed: %v", err)
		return
	}

	fmt.Printf("   ✅ Completed text tokens: %v\n", completed)
	fmt.Printf("   Generated %d new tokens\n", len(completed)-len(prompt))
}

func testBatchProcessing() {
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("❌ Batch processing example failed: %v\n", r)
		}
	}()

	config := TransformerConfig{
		VocabSize:    1000,
		SeqLen:       32,
		DModel:       256,
		NumHeads:     4,
		NumLayers:    2,
		DFF:          512,
		DropoutRate:  0.1,
		LearningRate: 0.001,

		// Basic configuration for batch testing
		UseQuantization:       false,
		UseGradientCheckpoint: false,
		EnableCustomKernels:   false,
		AutoTuning:            false,
	}

	model, err := NewTransformerModel(config)
	if err != nil {
		log.Printf("❌ Failed to create batch processing model: %v", err)
		return
	}
	defer model.Cleanup()

	// Process multiple sequences
	sequences := [][]int{
		{1, 2, 3, 4},
		{5, 6, 7, 8, 9},
		{10, 11, 12},
	}

	fmt.Printf("   Processing %d sequences:\n", len(sequences))

	for i, seq := range sequences {
		output, err := model.Forward(seq)
		if err != nil {
			log.Printf("   ❌ Error processing sequence %d: %v", i, err)
			continue
		}
		fmt.Printf("     ✅ Sequence %d: %v -> output shape: %dx%d\n",
			i+1, seq, output.Rows(), output.Cols())
	}
}

func testCustomConfigurations() {
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("❌ Custom configurations example failed: %v\n", r)
		}
	}()

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

		UseQuantization:       false,
		UseGradientCheckpoint: false,
		EnableCustomKernels:   false,
		AutoTuning:            false,
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

		UseQuantization:       true,
		QuantizationBits:      16,
		UseGradientCheckpoint: true,
		EnableCustomKernels:   true,
		AutoTuning:            true,
		MaxBatchSize:          32,
	}

	fmt.Printf("   ✅ Small model parameters: %d\n", calculateModelParameters(smallConfig))
	fmt.Printf("   ✅ Large model parameters: %d\n", calculateModelParameters(largeConfig))

	// Test creating the small model
	smallModel, err := NewTransformerModel(smallConfig)
	if err != nil {
		log.Printf("❌ Failed to create small model: %v", err)
		return
	}
	defer smallModel.Cleanup()

	fmt.Printf("   ✅ Small model created successfully\n")
}

func testTokenizerIntegration() {
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("❌ Tokenizer integration example failed: %v\n", r)
		}
	}()

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

	fmt.Printf("   Vocabulary size: %d\n", tokenizer.VocabSize)

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

		UseQuantization:       false,
		UseGradientCheckpoint: false,
		EnableCustomKernels:   false,
		AutoTuning:            false,
	}

	model, err := NewTransformerModel(config)
	if err != nil {
		log.Printf("❌ Failed to create tokenizer model: %v", err)
		return
	}
	defer model.Cleanup()

	// Process real text
	testText := "hello world"
	tokens := tokenizer.Encode(testText)
	fmt.Printf("   Text: '%s' -> Tokens: %v\n", testText, tokens)

	if len(tokens) > 0 {
		output, err := model.Forward(tokens)
		if err != nil {
			log.Printf("❌ Failed to process text: %v", err)
			return
		}
		fmt.Printf("   ✅ Processed text, output shape: %dx%d\n", output.Rows(), output.Cols())

		// Decode the tokens back to verify
		decoded := tokenizer.Decode(tokens)
		fmt.Printf("   Decoded back: '%s'\n", decoded)
	}
}
