package main

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strings"
	"unicode"
)

// Tokenizer handles text tokenization and encoding/decoding
type Tokenizer struct {
	VocabToID map[string]int
	IDToVocab map[int]string
	VocabSize int
	UNK_TOKEN string
	PAD_TOKEN string
	BOS_TOKEN string
	EOS_TOKEN string
	UNK_ID    int
	PAD_ID    int
	BOS_ID    int
	EOS_ID    int
}

// NewTokenizer creates a new tokenizer
func NewTokenizer() *Tokenizer {
	tokenizer := &Tokenizer{
		VocabToID: make(map[string]int),
		IDToVocab: make(map[int]string),
		UNK_TOKEN: "<UNK>",
		PAD_TOKEN: "<PAD>",
		BOS_TOKEN: "<BOS>",
		EOS_TOKEN: "<EOS>",
	}

	// Add special tokens
	tokenizer.addToken(tokenizer.PAD_TOKEN)
	tokenizer.addToken(tokenizer.UNK_TOKEN)
	tokenizer.addToken(tokenizer.BOS_TOKEN)
	tokenizer.addToken(tokenizer.EOS_TOKEN)

	tokenizer.PAD_ID = tokenizer.VocabToID[tokenizer.PAD_TOKEN]
	tokenizer.UNK_ID = tokenizer.VocabToID[tokenizer.UNK_TOKEN]
	tokenizer.BOS_ID = tokenizer.VocabToID[tokenizer.BOS_TOKEN]
	tokenizer.EOS_ID = tokenizer.VocabToID[tokenizer.EOS_TOKEN]

	return tokenizer
}

// addToken adds a token to the vocabulary
func (t *Tokenizer) addToken(token string) {
	if _, exists := t.VocabToID[token]; !exists {
		id := len(t.VocabToID)
		t.VocabToID[token] = id
		t.IDToVocab[id] = token
	}
}

// BuildVocabulary builds vocabulary from a list of texts
func (t *Tokenizer) BuildVocabulary(texts []string, maxVocabSize int) {
	fmt.Println("🔤 Building vocabulary...")

	// Count token frequencies
	tokenCounts := make(map[string]int)

	for _, text := range texts {
		tokens := t.tokenize(text)
		for _, token := range tokens {
			tokenCounts[token]++
		}
	}

	// Sort tokens by frequency (descending)
	type tokenFreq struct {
		token string
		count int
	}

	var sortedTokens []tokenFreq
	for token, count := range tokenCounts {
		sortedTokens = append(sortedTokens, tokenFreq{token, count})
	}

	sort.Slice(sortedTokens, func(i, j int) bool {
		return sortedTokens[i].count > sortedTokens[j].count
	})

	// Add tokens to vocabulary (up to maxVocabSize)
	added := 0
	for _, tf := range sortedTokens {
		if len(t.VocabToID) >= maxVocabSize {
			break
		}

		// Skip if already in vocabulary (special tokens)
		if _, exists := t.VocabToID[tf.token]; !exists {
			t.addToken(tf.token)
			added++
		}
	}

	t.VocabSize = len(t.VocabToID)

	fmt.Printf("✅ Vocabulary built: %d tokens (%d new tokens added)\n", t.VocabSize, added)
	fmt.Printf("   Most frequent tokens: ")
	for i := 0; i < 10 && i < len(sortedTokens); i++ {
		fmt.Printf("%s(%d) ", sortedTokens[i].token, sortedTokens[i].count)
	}
	fmt.Println()
}

// tokenize splits text into tokens (simple whitespace + punctuation tokenization)
func (t *Tokenizer) tokenize(text string) []string {
	// Convert to lowercase and clean
	text = strings.ToLower(strings.TrimSpace(text))

	var tokens []string
	var currentToken strings.Builder

	for _, r := range text {
		if unicode.IsSpace(r) {
			if currentToken.Len() > 0 {
				tokens = append(tokens, currentToken.String())
				currentToken.Reset()
			}
		} else if unicode.IsPunct(r) {
			if currentToken.Len() > 0 {
				tokens = append(tokens, currentToken.String())
				currentToken.Reset()
			}
			tokens = append(tokens, string(r))
		} else {
			currentToken.WriteRune(r)
		}
	}

	if currentToken.Len() > 0 {
		tokens = append(tokens, currentToken.String())
	}

	return tokens
}

// Encode converts text to token IDs
func (t *Tokenizer) Encode(text string) []int {
	tokens := t.tokenize(text)
	ids := make([]int, 0, len(tokens)+2) // +2 for BOS/EOS

	// Add BOS token
	ids = append(ids, t.BOS_ID)

	// Convert tokens to IDs
	for _, token := range tokens {
		if id, exists := t.VocabToID[token]; exists {
			ids = append(ids, id)
		} else {
			ids = append(ids, t.UNK_ID)
		}
	}

	// Add EOS token
	ids = append(ids, t.EOS_ID)

	return ids
}

// Decode converts token IDs back to text
func (t *Tokenizer) Decode(ids []int) string {
	var tokens []string

	for _, id := range ids {
		if token, exists := t.IDToVocab[id]; exists {
			// Skip special tokens in output
			if token != t.BOS_TOKEN && token != t.EOS_TOKEN && token != t.PAD_TOKEN {
				tokens = append(tokens, token)
			}
		}
	}

	return strings.Join(tokens, " ")
}

// EncodeSequences encodes multiple texts with padding
func (t *Tokenizer) EncodeSequences(texts []string, maxLength int) [][]int {
	sequences := make([][]int, len(texts))

	for i, text := range texts {
		ids := t.Encode(text)

		// Truncate if too long
		if len(ids) > maxLength {
			ids = ids[:maxLength]
			ids[maxLength-1] = t.EOS_ID // Ensure EOS at the end
		}

		// Pad if too short
		for len(ids) < maxLength {
			ids = append(ids, t.PAD_ID)
		}

		sequences[i] = ids
	}

	return sequences
}

// DatasetLoader handles loading and preprocessing datasets
type DatasetLoader struct {
	tokenizer *Tokenizer
}

// NewDatasetLoader creates a new dataset loader
func NewDatasetLoader(tokenizer *Tokenizer) *DatasetLoader {
	return &DatasetLoader{tokenizer: tokenizer}
}

// LoadTextFile loads text data from a file
func (dl *DatasetLoader) LoadTextFile(filename string) ([]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file %s: %v", filename, err)
	}
	defer file.Close()

	var texts []string
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if len(line) > 0 {
			texts = append(texts, line)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file %s: %v", filename, err)
	}

	return texts, nil
}

// CreateSampleDataset creates a sample dataset for testing
func (dl *DatasetLoader) CreateSampleDataset() []string {
	return []string{
		"The quick brown fox jumps over the lazy dog.",
		"Machine learning is a powerful tool for artificial intelligence.",
		"Natural language processing enables computers to understand human language.",
		"Deep learning models can learn complex patterns from data.",
		"Transformers have revolutionized the field of natural language processing.",
		"Attention mechanisms allow models to focus on relevant parts of the input.",
		"Large language models can generate coherent and contextually relevant text.",
		"Training neural networks requires careful tuning of hyperparameters.",
		"Data preprocessing is crucial for machine learning success.",
		"The future of AI looks bright with continued research and development.",
		"Programming in Go is efficient and straightforward.",
		"CUDA enables parallel computing on graphics processing units.",
		"Optimization algorithms help minimize loss functions during training.",
		"Gradient descent is a fundamental optimization technique.",
		"Backpropagation computes gradients for neural network training.",
		"Overfitting occurs when models memorize training data too closely.",
		"Regularization techniques help prevent overfitting in machine learning.",
		"Cross-validation is important for evaluating model performance.",
		"Feature engineering can significantly improve model accuracy.",
		"Ensemble methods combine multiple models for better predictions.",
	}
}

// PrepareTrainingData prepares data for training
func (dl *DatasetLoader) PrepareTrainingData(texts []string, maxSeqLen int) ([][]int, error) {
	fmt.Printf("📊 Preparing training data with max sequence length: %d\n", maxSeqLen)

	// Encode all texts
	sequences := dl.tokenizer.EncodeSequences(texts, maxSeqLen)

	// Filter out sequences that are mostly padding
	var validSequences [][]int
	minValidTokens := 5 // Minimum number of non-padding tokens

	for _, seq := range sequences {
		validTokens := 0
		for _, id := range seq {
			if id != dl.tokenizer.PAD_ID {
				validTokens++
			}
		}

		if validTokens >= minValidTokens {
			validSequences = append(validSequences, seq)
		}
	}

	fmt.Printf("✅ Prepared %d valid sequences (filtered from %d total)\n",
		len(validSequences), len(sequences))

	return validSequences, nil
}

// SaveVocabulary saves the vocabulary to a file
func (t *Tokenizer) SaveVocabulary(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create vocabulary file: %v", err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	defer writer.Flush()

	// Write vocabulary in order of IDs
	for i := 0; i < t.VocabSize; i++ {
		if token, exists := t.IDToVocab[i]; exists {
			_, err := writer.WriteString(fmt.Sprintf("%d\t%s\n", i, token))
			if err != nil {
				return fmt.Errorf("failed to write vocabulary: %v", err)
			}
		}
	}

	return nil
}

// LoadVocabulary loads vocabulary from a file
func (t *Tokenizer) LoadVocabulary(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open vocabulary file: %v", err)
	}
	defer file.Close()

	// Clear existing vocabulary
	t.VocabToID = make(map[string]int)
	t.IDToVocab = make(map[int]string)

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, "\t")
		if len(parts) != 2 {
			continue
		}

		var id int
		if _, err := fmt.Sscanf(parts[0], "%d", &id); err != nil {
			continue
		}

		token := parts[1]
		t.VocabToID[token] = id
		t.IDToVocab[id] = token
	}

	t.VocabSize = len(t.VocabToID)

	// Update special token IDs
	if id, exists := t.VocabToID[t.PAD_TOKEN]; exists {
		t.PAD_ID = id
	}
	if id, exists := t.VocabToID[t.UNK_TOKEN]; exists {
		t.UNK_ID = id
	}
	if id, exists := t.VocabToID[t.BOS_TOKEN]; exists {
		t.BOS_ID = id
	}
	if id, exists := t.VocabToID[t.EOS_TOKEN]; exists {
		t.EOS_ID = id
	}

	return scanner.Err()
}

// GetVocabularyStats returns statistics about the vocabulary
func (t *Tokenizer) GetVocabularyStats() map[string]interface{} {
	return map[string]interface{}{
		"vocab_size": t.VocabSize,
		"pad_id":     t.PAD_ID,
		"unk_id":     t.UNK_ID,
		"bos_id":     t.BOS_ID,
		"eos_id":     t.EOS_ID,
		"pad_token":  t.PAD_TOKEN,
		"unk_token":  t.UNK_TOKEN,
		"bos_token":  t.BOS_TOKEN,
		"eos_token":  t.EOS_TOKEN,
	}
}
