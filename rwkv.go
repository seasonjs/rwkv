// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

package rwkv

import (
	"errors"
	"os"
	"strings"
)

type RwkvModel struct {
	cRwkv      CRwkv
	tokenizer  Tokenizer
	dylibPath  string
	ctx        *RwkvCtx
	options    *RwkvOptions
	isAutoLoad bool
}

type RwkvOptions struct {
	PrintError    bool
	MaxTokens     int
	StopString    string
	Temperature   float32
	TopP          float32
	TokenizerType TokenizerType
}

func hasCtx(ctx *RwkvCtx) error {
	if ctx.ctx == 0 {
		return RwkvErrors(RwkvErrorCtx)
	}
	return nil
}

func NewRwkvAutoModel(options RwkvOptions) (*RwkvModel, error) {
	file, err := dumpRwkvLibrary()
	if err != nil {
		return nil, err
	}

	dylibPath := file.Name()

	model, err := NewRwkvModel(dylibPath, options)
	if err != nil {
		return nil, err
	}
	model.isAutoLoad = true
	return model, nil
}

func NewRwkvModel(dylibPath string, options RwkvOptions) (*RwkvModel, error) {
	cRwkv, err := NewCRwkv(dylibPath)
	if err != nil {
		return nil, err
	}

	var tk Tokenizer

	if options.TokenizerType == Normal {
		tk, err = NewNormalTokenizer()
	}

	if options.TokenizerType == World {
		tk, err = NewWorldTokenizer()
	}

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

func (m *RwkvModel) LoadFromFile(path string, thread uint32) error {
	_, err := os.Stat(path)
	if err != nil {
		return errors.New("the system cannot find the model file specified")
	}
	ctx := m.cRwkv.RwkvInitFromFile(path, thread)
	m.ctx = ctx
	// by default disable error printing and handle errors by go error
	m.cRwkv.RwkvSetPrintErrors(ctx, m.options.PrintError)
	return nil
}

func (m *RwkvModel) QuantizeModelFile(in, out string, format QuantizedFormat) error {
	return m.cRwkv.RwkvQuantizeModelFile(m.ctx, in, out, format)
}

func (m *RwkvModel) Close() error {
	if m.ctx != nil {
		if err := m.cRwkv.RwkvFree(m.ctx); err != nil {
			return err
		}
		m.ctx = nil
	}
	if m.isAutoLoad {
		err := os.Remove(m.dylibPath)
		return err
	}

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
	encode, err := s.rwkvModel.tokenizer.Encode(input)
	if err != nil {
		return err
	}
	for _, token := range encode {
		err = s.rwkvModel.cRwkv.RwkvEval(s.rwkvModel.ctx, uint32(token), s.state, s.state, s.logits)
		if err != nil {
			return err
		}
	}
	return nil
}

func (s *RwkvState) generateResponse(callback func(s string) bool) (string, error) {
	responseText := ""
	for i := 0; i < s.rwkvModel.options.MaxTokens; i++ {

		token, err := SampleLogits(s.logits, s.rwkvModel.options.Temperature, s.rwkvModel.options.TopP, map[int]float32{})
		if err != nil {
			return "", err
		}

		err = s.rwkvModel.cRwkv.RwkvEval(s.rwkvModel.ctx, uint32(token), s.state, s.state, s.logits)
		if err != nil {
			return "", err
		}

		chars := s.rwkvModel.tokenizer.Decode([]int{token})
		responseText += chars
		if callback != nil && !callback(chars) {
			break
		}
		if strings.Contains(responseText, s.rwkvModel.options.StopString) {
			responseText = strings.Split(responseText, s.rwkvModel.options.StopString)[0]
			break
		}
	}
	return responseText, nil
}
