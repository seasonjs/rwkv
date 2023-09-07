// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

package rwkv

import (
	"github.com/sugarme/tokenizer"
	"strings"
)

type RwkvModel struct {
	cRwkv     CRwkv
	tokenizer *tokenizer.Tokenizer
	dylibPath string
	ctx       *RwkvCtx
	options   *RwkvOptions
}

type RwkvOptions struct {
	printError  bool
	maxTokens   int
	stopString  string
	temperature float32
	topP        float32
}

func hasCtx(ctx *RwkvCtx) error {
	if ctx.ctx == 0 {
		return RwkvErrors(RwkvErrorCtx)
	}
	return nil
}

func NewRwkvModel(dylibPath string, options RwkvOptions) (*RwkvModel, error) {
	cRwkv, err := NewCRwkv(dylibPath)
	if err != nil {
		return nil, err
	}

	// TODO: support word tokenizer
	tk, err := NormalTokenizer()
	if err != nil {
		return nil, err
	}

	return &RwkvModel{
		dylibPath: dylibPath,
		cRwkv:     cRwkv,
		options:   &options,
		tokenizer: tk,
	}, nil
}

func (m *RwkvModel) LoadFromFile(path string, thread uint32) {
	ctx := m.cRwkv.RwkvInitFromFile(path, thread)
	m.ctx = ctx
	// by default disable error printing and handle errors by go error
	m.cRwkv.RwkvSetPrintErrors(ctx, m.options.printError)
}

func (m *RwkvModel) Close() error {
	if err := hasCtx(m.ctx); err != nil {
		return err
	}
	m.cRwkv.RwkvFree(m.ctx)
	m.ctx = nil
	return nil
}

type RwkvState struct {
	state     []float32
	logits    []float32
	rwkvModel *RwkvModel
}

// InitState give a new state for new chat context state
func (m *RwkvModel) InitState() (*RwkvState, error) {
	if err := hasCtx(m.ctx); err != nil {
		return nil, err
	}
	state := make([]float32, m.cRwkv.RwkvGetStateLength(m.ctx))
	m.cRwkv.RwkvInitState(m.ctx, state)
	logits := make([]float32, m.cRwkv.RwkvGetLogitsLength(m.ctx))
	return &RwkvState{
		state:     state,
		rwkvModel: m,
		logits:    logits,
	}, nil
}

// Predict give current chat a response
func (s *RwkvState) Predict(input string) (string, error) {
	err := s.handelInput(input)
	if err != nil {
		return "", err
	}
	return s.generateResponse(nil)
}

func (s *RwkvState) PredictStream(input string, output chan string) {
	go func() {
		err := s.handelInput(input)
		if err != nil {
			output <- err.Error()
			close(output)
			return
		}
		_, err = s.generateResponse(func(s string) bool {
			output <- s
			return true
		})
		close(output)
	}()
}

func (s *RwkvState) handelInput(input string) error {
	in := tokenizer.NewSingleEncodeInput(tokenizer.NewInputSequence(input))
	encode, err := s.rwkvModel.tokenizer.Encode(in, false)
	if err != nil {
		return err
	}
	for _, token := range encode.Ids {
		err = s.rwkvModel.cRwkv.RwkvEval(s.rwkvModel.ctx, uint32(token), s.state, s.state, s.logits)
		if err != nil {
			return err
		}
	}
	return nil
}

func (s *RwkvState) generateResponse(callback func(s string) bool) (string, error) {
	responseText := ""
	for i := 0; i < s.rwkvModel.options.maxTokens; i++ {

		token, err := SampleLogits(s.logits, s.rwkvModel.options.temperature, s.rwkvModel.options.topP, map[int]float32{})
		if err != nil {
			return "", err
		}

		err = s.rwkvModel.cRwkv.RwkvEval(s.rwkvModel.ctx, uint32(token), s.state, s.state, s.logits)
		if err != nil {
			return "", err
		}

		chars := s.rwkvModel.tokenizer.Decode([]int{token}, false)
		responseText += chars
		if callback != nil && !callback(chars) {
			break
		}
		if strings.Contains(responseText, s.rwkvModel.options.stopString) {
			responseText = strings.Split(responseText, s.rwkvModel.options.stopString)[0]
			break
		}
	}
	return responseText, nil
}
