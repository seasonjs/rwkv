// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

package rwkv

import "github.com/sugarme/tokenizer"

type Model interface {
	LoadFromFile(path string, thread uint32)
	Close() error
}

type RwkvModel struct {
	cRwkv     CRwkv
	tokenizer *tokenizer.Tokenizer
	dylibPath string
	ctx       RwkvCtx
	options   *RwkvOptions
}

type RwkvOptions struct {
	printError  bool
	maxTokens   int
	stopString  string
	temperature float32
	topP        float32
}

func hasCtx(ctx RwkvCtx) error {
	if ctx.ctx == -1 {
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

//func (m *RwkvModel) Predict(input string) (string, error) {
//
//}
//
//func (m *RwkvModel) PredictStream() (string, error) {
//
//}

func (m *RwkvModel) Close() error {
	if err := hasCtx(m.ctx); err != nil {
		return err
	}
	m.cRwkv.RwkvFree(m.ctx)
	m.ctx = RwkvCtx{ctx: -1}
	return nil
}
