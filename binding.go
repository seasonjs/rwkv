// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

package rwkv

import (
	"github.com/ebitengine/purego"
)

type QuantizedFormat string

const (
	Q4_0 QuantizedFormat = "Q4_0"
	Q4_1 QuantizedFormat = "Q4_1"
	Q5_0 QuantizedFormat = "Q5_0"
	Q5_1 QuantizedFormat = "Q5_0"
	Q8_0 QuantizedFormat = "Q8_0"
)

const cRwkvSetPrintErrors = "rwkv_set_print_errors"
const cRwkvGetPrintErrors = "rwkv_get_print_errors"
const cRwkvGetLastError = "rwkv_get_last_error"
const cRwkvInitFromFile = "rwkv_init_from_file"

const cRwkvCloneContext = "rwkv_clone_context"
const cRwkvGpuOffloadLayers = "rwkv_gpu_offload_layers"
const cRwkvEval = "rwkv_eval"
const cRwkvEvalSequence = "rwkv_eval_sequence"

const cRwkvGetNVocab = "rwkv_get_n_vocab"
const cRwkvGetNEmbedding = "rwkv_get_n_embed"
const cRwkvGetNLayer = "rwkv_get_n_layer"
const cRwkvGetStateLength = "rwkv_get_state_len"

const cRwkvGetLogitsLength = "rwkv_get_logits_len"
const cRwkvInitState = "rwkv_init_state"
const cRwkvFree = "rwkv_free"
const cRwkvQuantizeModelFile = "rwkv_quantize_model_file"

const cRwkvGetSystemInfoString = "rwkv_get_system_info_string"

type RwkvCtx struct {
	ctx uintptr
}
type CRwkv interface {
	// RwkvSetPrintErrors Sets whether errors are automatically printed to stderr.
	// If this is set to false, you are responsible for calling rwkv_last_error manually if an operation fails.
	// - ctx: the context to suppress error messages for.
	//   If NULL, affects model load (rwkv_init_from_file) and quantization (rwkv_quantize_model_file) errors,
	//   as well as the default for new context.
	// - print_errors: whether error messages should be automatically printed.
	RwkvSetPrintErrors(ctx RwkvCtx, enable bool)

	// RwkvGetPrintErrors Gets whether errors are automatically printed to stderr.
	// - ctx: the context to retrieve the setting for, or NULL for the global setting.
	RwkvGetPrintErrors(ctx RwkvCtx) bool

	// RwkvGetLastError Retrieves and clears the error flags.
	// - ctx: the context the retrieve the error for, or NULL for the global error.
	RwkvGetLastError(ctx RwkvCtx) error

	// RwkvInitFromFile Loads the model from a file and prepares it for inference.
	// Returns NULL on any error.
	// - model_file_path: path to model file in ggml format.
	// - n_threads: count of threads to use, must be positive.
	RwkvInitFromFile(filePath string, threads uint32) RwkvCtx

	// RwkvCloneContext Creates a new context from an existing one.
	// This can allow you to run multiple rwkv_eval's in parallel, without having to load a single model multiple times.
	// Each rwkv_context can have one eval running at a time.
	// Every rwkv_context must be freed using rwkv_free.
	// - ctx: context to be cloned.
	// - n_threads: count of threads to use, must be positive.
	RwkvCloneContext(ctx RwkvCtx, threads uint32) RwkvCtx

	// RwkvGpuOffloadLayers Offloads specified layers of context onto GPU using cuBLAS, if it is enabled.
	// If rwkv.cpp was compiled without cuBLAS support, this function is a no-op.
	RwkvGpuOffloadLayers(ctx RwkvCtx, nGpuLayers uint32) error

	// RwkvEval Evaluates the model for a single token.
	// Not thread-safe. For parallel inference, call rwkv_clone_context to create one rwkv_context for each thread.
	// Returns false on any error.
	// - token: next token index, in range 0 <= token < n_vocab.
	// - state_in: FP32 buffer of size rwkv_get_state_len(); or NULL, if this is a first pass.
	// - state_out: FP32 buffer of size rwkv_get_state_len(). This buffer will be written to if non-NULL.
	// - logits_out: FP32 buffer of size rwkv_get_logits_len(). This buffer will be written to if non-NULL.
	RwkvEval(ctx RwkvCtx, token uint32, stateIn []float32, stateOut []float32, logitsOut []float32) error

	// RwkvEvalSequence Evaluates the model for a sequence of tokens.
	// Uses a faster algorithm than rwkv_eval if you do not need the state and logits for every token. Best used with batch sizes of 64 or so.
	// Has to build a computation graph on the first call for a given sequence, but will use this cached graph for subsequent calls of the same sequence length.
	// Not thread-safe. For parallel inference, call rwkv_clone_context to create one rwkv_context for each thread.
	// Returns false on any error.
	// - tokens: pointer to an array of tokens. If NULL, the graph will be built and cached, but not executed: this can be useful for initialization.
	// - sequence_len: number of tokens to read from the array.
	// - state_in: FP32 buffer of size rwkv_get_state_len(), or NULL if this is a first pass.
	// - state_out: FP32 buffer of size rwkv_get_state_len(). This buffer will be written to if non-NULL.
	// - logits_out: FP32 buffer of size rwkv_get_logits_len(). This buffer will be written to if non-NULL.
	RwkvEvalSequence(ctx RwkvCtx, token uint32, sequenceLen uint64, stateIn []float32, stateOut []float32, logitsOut []float32) error

	// RwkvGetNVocab Returns the number of tokens in the given model's vocabulary.
	// Useful for telling 20B_tokenizer models (n_vocab = 50277) apart from World models (n_vocab = 65536).
	RwkvGetNVocab(ctx RwkvCtx) uint64

	// RwkvGetNEmbedding Returns the number of elements in the given model's embedding.
	// Useful for reading individual fields of a model's hidden state.
	RwkvGetNEmbedding(ctx RwkvCtx) uint64

	// RwkvGetNLayer Returns the number of layers in the given model.
	// Useful for always offloading the entire model to GPU.
	RwkvGetNLayer(ctx RwkvCtx) uint64

	// RwkvGetStateLength Returns the number of float elements in a complete state for the given model.
	// This is the number of elements you'll need to allocate for a call to rwkv_eval, rwkv_eval_sequence, or rwkv_init_state.
	RwkvGetStateLength(ctx RwkvCtx) uint64

	// RwkvGetLogitsLength Returns the number of float elements in the logits output of a given model.
	// This is currently always identical to n_vocab.
	RwkvGetLogitsLength(ctx RwkvCtx) uint64

	// RwkvInitState Initializes the given state so that passing it to rwkv_eval or rwkv_eval_sequence would be identical to passing NULL.
	// Useful in cases where tracking the first call to these functions may be annoying or expensive.
	// State must be initialized for behavior to be defined, passing a zeroed state to rwkv.cpp functions will result in NaNs.
	// - state: FP32 buffer of size rwkv_get_state_len() to initialize
	RwkvInitState(ctx RwkvCtx, state []float32)

	// RwkvFree Frees all allocated memory and the context.
	// Does not need to be called on the same thread that created the rwkv_context.
	RwkvFree(ctx RwkvCtx)

	// RwkvQuantizeModelFile Quantizes FP32 or FP16 model to one of quantized formats.
	// Returns false on any error. Error messages would be printed to stderr.
	// - model_file_path_in: path to model file in ggml format, must be either FP32 or FP16.
	// - model_file_path_out: quantized model will be written here.
	// - format_name: must be one of available format names below.
	// Available format names:
	// - Q4_0
	// - Q4_1
	// - Q5_0
	// - Q5_1
	// - Q8_0
	RwkvQuantizeModelFile(ctx RwkvCtx, in, out string, format QuantizedFormat) error

	// RwkvGetSystemInfoString Returns system information string.
	RwkvGetSystemInfoString() string
}

type CRwkvImpl struct {
	libRwkv                  uintptr
	cRwkvSetPrintErrors      func(uintptr, bool)
	cRwkvGetPrintErrors      func(uintptr) bool
	cRwkvGetLastError        func(uintptr) uint32
	cRwkvInitFromFile        func(modelFilePath string, nThreads uint32) uintptr
	cRwkvCloneContext        func(ctx uintptr, nThreads uint32) uintptr
	cRwkvGpuOffloadLayers    func(ctx uintptr, nGpuLayers uint32) bool
	cRwkvEval                func(ctx uintptr, token uint32, stateIn []float32, stateOut []float32, logitsOut []float32) bool
	cRwkvEvalSequence        func(ctx uintptr, token uint32, sequenceLen uint64, stateIn []float32, stateOut []float32, logitsOut []float32) bool
	cRwkvGetNVocab           func(ctx uintptr) uint64
	cRwkvGetNEmbedding       func(ctx uintptr) uint64
	cRwkvGetNLayer           func(ctx uintptr) uint64
	cRwkvGetStateLength      func(ctx uintptr) uint64
	cRwkvGetLogitsLength     func(ctx uintptr) uint64
	cRwkvInitState           func(ctx uintptr, state []float32)
	cRwkvFree                func(ctx uintptr)
	cRwkvQuantizeModelFile   func(modelFilePathIn string, modelFilePathOut string, formatName string) bool
	cRwkvGetSystemInfoString func() string
}

func NewCRwkv() (*CRwkvImpl, error) {
	libRwkv, err := openLibrary(libraryPath)
	if err != nil {
		return nil, err
	}
	var (
		rwkvSetPrintErrors      func(uintptr, bool)
		rwkvGetPrintErrors      func(uintptr) bool
		rwkvGetLastError        func(uintptr) uint32
		rwkvInitFromFile        func(modelFilePath string, nThreads uint32) uintptr
		rwkvCloneContext        func(ctx uintptr, nThreads uint32) uintptr
		rwkvGpuOffloadLayers    func(ctx uintptr, nGpuLayers uint32) bool
		rwkvEval                func(ctx uintptr, token uint32, stateIn []float32, stateOut []float32, logitsOut []float32) bool
		rwkvEvalSequence        func(ctx uintptr, token uint32, sequenceLen uint64, stateIn []float32, stateOut []float32, logitsOut []float32) bool
		rwkvGetNVocab           func(ctx uintptr) uint64
		rwkvGetNEmbedding       func(ctx uintptr) uint64
		rwkvGetNLayer           func(ctx uintptr) uint64
		rwkvGetStateLength      func(ctx uintptr) uint64
		rwkvGetLogitsLength     func(ctx uintptr) uint64
		rwkvInitState           func(ctx uintptr, state []float32)
		rwkvFree                func(ctx uintptr)
		rwkvQuantizeModelFile   func(modelFilePathIn string, modelFilePathOut string, formatName string) bool
		rwkvGetSystemInfoString func() string
	)
	purego.RegisterLibFunc(&rwkvSetPrintErrors, libRwkv, cRwkvSetPrintErrors)
	purego.RegisterLibFunc(&rwkvGetPrintErrors, libRwkv, cRwkvGetPrintErrors)
	purego.RegisterLibFunc(&rwkvGetLastError, libRwkv, cRwkvGetLastError)
	purego.RegisterLibFunc(&rwkvInitFromFile, libRwkv, cRwkvInitFromFile)

	purego.RegisterLibFunc(&rwkvCloneContext, libRwkv, cRwkvCloneContext)
	purego.RegisterLibFunc(&rwkvGpuOffloadLayers, libRwkv, cRwkvGpuOffloadLayers)
	purego.RegisterLibFunc(&rwkvEval, libRwkv, cRwkvEval)
	purego.RegisterLibFunc(&rwkvEvalSequence, libRwkv, cRwkvEvalSequence)

	purego.RegisterLibFunc(&rwkvGetNVocab, libRwkv, cRwkvGetNVocab)
	purego.RegisterLibFunc(&rwkvGetNEmbedding, libRwkv, cRwkvGetNEmbedding)
	purego.RegisterLibFunc(&rwkvGetNLayer, libRwkv, cRwkvGetNLayer)
	purego.RegisterLibFunc(&rwkvGetStateLength, libRwkv, cRwkvGetStateLength)

	purego.RegisterLibFunc(&rwkvGetLogitsLength, libRwkv, cRwkvGetLogitsLength)
	purego.RegisterLibFunc(&rwkvInitState, libRwkv, cRwkvInitState)
	purego.RegisterLibFunc(&rwkvFree, libRwkv, cRwkvFree)
	purego.RegisterLibFunc(&rwkvQuantizeModelFile, libRwkv, cRwkvQuantizeModelFile)

	purego.RegisterLibFunc(&rwkvGetSystemInfoString, libRwkv, cRwkvGetSystemInfoString)

	return &CRwkvImpl{
		libRwkv: libRwkv,

		cRwkvSetPrintErrors: rwkvSetPrintErrors,
		cRwkvGetPrintErrors: rwkvGetPrintErrors,
		cRwkvGetLastError:   rwkvGetLastError,
		cRwkvInitFromFile:   rwkvInitFromFile,

		cRwkvCloneContext:     rwkvCloneContext,
		cRwkvGpuOffloadLayers: rwkvGpuOffloadLayers,
		cRwkvEval:             rwkvEval,
		cRwkvEvalSequence:     rwkvEvalSequence,

		cRwkvGetNVocab:      rwkvGetNVocab,
		cRwkvGetNEmbedding:  rwkvGetNEmbedding,
		cRwkvGetNLayer:      rwkvGetNLayer,
		cRwkvGetStateLength: rwkvGetStateLength,

		cRwkvGetLogitsLength:   rwkvGetLogitsLength,
		cRwkvInitState:         rwkvInitState,
		cRwkvFree:              rwkvFree,
		cRwkvQuantizeModelFile: rwkvQuantizeModelFile,

		cRwkvGetSystemInfoString: rwkvGetSystemInfoString,
	}, nil
}
func (c *CRwkvImpl) RwkvSetPrintErrors(ctx RwkvCtx, enable bool) {
	c.cRwkvSetPrintErrors(ctx.ctx, enable)
}
func (c *CRwkvImpl) RwkvGetPrintErrors(ctx RwkvCtx) bool {
	return c.cRwkvGetPrintErrors(ctx.ctx)
}

func (c *CRwkvImpl) RwkvGetLastError(ctx RwkvCtx) error {
	cErr := c.cRwkvGetLastError(ctx.ctx)
	err := RwkvErrors(cErr)
	if err == RwkvErrorNone {
		return nil
	}
	return err
}

func (c *CRwkvImpl) RwkvInitFromFile(filePath string, threads uint32) RwkvCtx {
	ctx := c.cRwkvInitFromFile(filePath, threads)
	return RwkvCtx{ctx: ctx}
}

func (c *CRwkvImpl) RwkvCloneContext(ctx RwkvCtx, threads uint32) RwkvCtx {
	newCtx := c.cRwkvCloneContext(ctx.ctx, threads)
	return RwkvCtx{ctx: newCtx}
}

func (c *CRwkvImpl) RwkvGpuOffloadLayers(ctx RwkvCtx, nGpuLayers uint32) error {
	ok := c.cRwkvGpuOffloadLayers(ctx.ctx, nGpuLayers)
	if !ok {
		return c.RwkvGetLastError(ctx)
	}
	return nil
}

func (c *CRwkvImpl) RwkvEval(ctx RwkvCtx, token uint32, stateIn []float32, stateOut []float32, logitsOut []float32) error {
	ok := c.cRwkvEval(ctx.ctx, token, stateIn, stateOut, logitsOut)
	if !ok {
		return c.RwkvGetLastError(ctx)
	}
	return nil
}

func (c *CRwkvImpl) RwkvEvalSequence(ctx RwkvCtx, token uint32, sequenceLen uint64, stateIn []float32, stateOut []float32, logitsOut []float32) error {
	ok := c.cRwkvEvalSequence(ctx.ctx, token, sequenceLen, stateIn, stateOut, logitsOut)
	if !ok {
		return c.RwkvGetLastError(ctx)
	}
	return nil
}

func (c *CRwkvImpl) RwkvGetNVocab(ctx RwkvCtx) uint64 {
	return c.cRwkvGetNVocab(ctx.ctx)
}

func (c *CRwkvImpl) RwkvGetNEmbedding(ctx RwkvCtx) uint64 {
	return c.cRwkvGetNEmbedding(ctx.ctx)
}

func (c *CRwkvImpl) RwkvGetNLayer(ctx RwkvCtx) uint64 {
	return c.cRwkvGetNLayer(ctx.ctx)
}

func (c *CRwkvImpl) RwkvGetStateLength(ctx RwkvCtx) uint64 {
	return c.cRwkvGetStateLength(ctx.ctx)
}

func (c *CRwkvImpl) RwkvGetLogitsLength(ctx RwkvCtx) uint64 {
	return c.cRwkvGetLogitsLength(ctx.ctx)
}

func (c *CRwkvImpl) RwkvInitState(ctx RwkvCtx, state []float32) {
	c.cRwkvInitState(ctx.ctx, state)
}

func (c *CRwkvImpl) RwkvFree(ctx RwkvCtx) {
	c.cRwkvFree(ctx.ctx)
}

func (c *CRwkvImpl) RwkvQuantizeModelFile(ctx RwkvCtx, in, out string, format QuantizedFormat) error {
	ok := c.cRwkvQuantizeModelFile(in, out, string(format))
	if !ok {
		return c.RwkvGetLastError(ctx)
	}
	return nil
}

func (c *CRwkvImpl) RwkvGetSystemInfoString() string {
	return c.cRwkvGetSystemInfoString()
}
