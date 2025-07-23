package main

import (
	"fmt"
	"log"
)

// Quick demo of transformer usage
func QuickDemo() {
	fmt.Println("🎯 Quick Transformer Demo")
	fmt.Println("=========================")

	// 1. Create a small model for testing
	config := TransformerConfig{
		VocabSize:    500, // Small vocabulary
		SeqLen:       32,  // Short sequences
		DModel:       128, // Small embedding
		NumHeads:     4,   // Fewer heads
		NumLayers:    2,   // Fewer layers
		DFF:          256, // Smaller feed-forward
		DropoutRate:  0.1,
		LearningRate: 0.001,
	}

	fmt.Printf("Model will have %d parameters\n", calculateModelParameters(config))

	// 2. Create the model
	model, err := NewTransformerModel(config)
	if err != nil {
		log.Fatal("Error creating model:", err)
	}
	defer model.Cleanup()

	// 3. Process some tokens
	fmt.Println("\n--- Forward Pass ---")
	inputTokens := []int{1, 5, 10, 25}
	fmt.Printf("Input tokens: %v\n", inputTokens)

	output, err := model.Forward(inputTokens)
	if err != nil {
		log.Fatal("Forward pass error:", err)
	}
	fmt.Printf("Output shape: %dx%d (sequence_length x vocab_size)\n",
		output.Rows(), output.Cols())

	// 4. Generate text
	fmt.Println("\n--- Text Generation ---")
	prompt := []int{1, 2}
	fmt.Printf("Prompt: %v\n", prompt)

	generated, err := model.Generate(prompt, 8, 0.8)
	if err != nil {
		log.Fatal("Generation error:", err)
	}
	fmt.Printf("Generated: %v\n", generated)
	fmt.Printf("Added %d new tokens\n", len(generated)-len(prompt))

	// 5. Working with tokenizer
	fmt.Println("\n--- With Real Text ---")
	tokenizer := NewTokenizer()

	// Build vocabulary from sample text
	texts := []string{
		"hello world", "good morning", "how are you",
		"machine learning", "transformer model",
	}
	tokenizer.BuildVocabulary(texts, 200)

	testText := "hello world"
	tokens := tokenizer.Encode(testText)
	fmt.Printf("Text: '%s'\n", testText)
	fmt.Printf("Tokens: %v\n", tokens)
	fmt.Printf("Decoded: '%s'\n", tokenizer.Decode(tokens))

	if len(tokens) > 0 {
		output, err := model.Forward(tokens)
		if err == nil {
			fmt.Printf("Processed successfully: %dx%d\n", output.Rows(), output.Cols())
		}
	}

	fmt.Println("\n✅ Demo completed!")
}

// Uncomment this function call in main() to run the demo
// func main() {
//     QuickDemo()
// }
