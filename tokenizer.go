// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

package rwkv

import (
	"embed"
	"encoding/json"
	"fmt"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

type TokenizerType uint8

const (
	Normal TokenizerType = iota
	World
)

type Tokenizer interface {
	Encode(in string) ([]int, error)
	Decode(in []int) string
}

//go:embed 20B_tokenizer.json
var tokenizerFS embed.FS

type NormalTokenizer struct {
	tk *tokenizer.Tokenizer
}

func NewNormalTokenizer() (*NormalTokenizer, error) {
	f, err := tokenizerFS.Open("20B_tokenizer.json")
	if err != nil {
		return nil, err
	}

	dec := json.NewDecoder(f)

	var config *tokenizer.Config
	err = dec.Decode(&config)
	if err != nil {
		return nil, err
	}

	model, err := pretrained.CreateModel(config)
	if err != nil {
		err := fmt.Errorf("creating Model failed: %v", err)
		return nil, err
	}
	tk := tokenizer.NewTokenizer(model)

	// 2. Normalizer
	n, err := pretrained.CreateNormalizer(config.Normalizer)
	if err != nil {
		err = fmt.Errorf("creating Normalizer failed: %v", err)
		return nil, err
	}
	tk.WithNormalizer(n)

	// 3. PreTokenizer
	preTok, err := pretrained.CreatePreTokenizer(config.PreTokenizer)
	if err != nil {
		err = fmt.Errorf("creating PreTokenizer failed: %v", err)
		return nil, err
	}
	tk.WithPreTokenizer(preTok)

	// 4. PostProcessor
	postProcessor, err := pretrained.CreatePostProcessor(config.PostProcessor)
	if err != nil {
		err = fmt.Errorf("creating PostProcessor failed: %v", err)
		return nil, err
	}
	tk.WithPostProcessor(postProcessor)

	// 5. Decoder
	decoder, err := pretrained.CreateDecoder(config.Decoder)
	if err != nil {
		err = fmt.Errorf("creating Decoder failed: %v", err)
		return nil, err
	}
	tk.WithDecoder(decoder)

	// 6. AddedVocabulary
	specialAddedTokens, addedTokens := pretrained.CreateAddedTokens(config.AddedTokens)
	if len(specialAddedTokens) > 0 {
		tk.AddSpecialTokens(specialAddedTokens)
	}
	if len(addedTokens) > 0 {
		tk.AddTokens(addedTokens)
	}

	// 7. TruncationParams
	truncParams, err := pretrained.CreateTruncationParams(config.Truncation)
	if err != nil {
		err = fmt.Errorf("creating TruncationParams failed: %v", err)
		return nil, err
	}
	tk.WithTruncation(truncParams)

	// 8. PaddingParams
	paddingParams, err := pretrained.CreatePaddingParams(config.Padding)
	if err != nil {
		err = fmt.Errorf("creating PaddingParams failed: %v", err)
		return nil, err
	}
	tk.WithPadding(paddingParams)

	return &NormalTokenizer{tk: tk}, nil
}

func (t *NormalTokenizer) Encode(input string) ([]int, error) {
	in := tokenizer.NewSingleEncodeInput(tokenizer.NewInputSequence(input))
	encode, err := t.tk.Encode(in, false)
	if err != nil {
		return nil, err
	}
	return encode.Ids, nil
}

func (t *NormalTokenizer) Decode(ids []int) string {
	out := t.tk.Decode(ids, false)
	return out
}
