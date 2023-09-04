// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

package rwkv

type RwkvQuantizedFormat string

const (
	Q4_0 RwkvQuantizedFormat = "Q4_0"
	Q4_1 RwkvQuantizedFormat = "Q4_1"
	Q5_0 RwkvQuantizedFormat = "Q5_0"
	Q5_1 RwkvQuantizedFormat = "Q5_0"
	Q8_0 RwkvQuantizedFormat = "Q8_0"
)

type RwkvCtx struct {
	ctx uintptr
}
type Rwkv interface {
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
	RwkvGetLastError(ctx RwkvCtx) RwkvErrors

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
	RwkvGpuOffloadLayers(ctx RwkvCtx, nGpuLayers uint32) bool

	// RwkvEval Evaluates the model for a single token.
	// Not thread-safe. For parallel inference, call rwkv_clone_context to create one rwkv_context for each thread.
	// Returns false on any error.
	// - token: next token index, in range 0 <= token < n_vocab.
	// - state_in: FP32 buffer of size rwkv_get_state_len(); or NULL, if this is a first pass.
	// - state_out: FP32 buffer of size rwkv_get_state_len(). This buffer will be written to if non-NULL.
	// - logits_out: FP32 buffer of size rwkv_get_logits_len(). This buffer will be written to if non-NULL.
	RwkvEval(modelFilePathIn, modelFilePathOut string, formatName string) (bool, error)

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
	RwkvEvalSequence()

	// RwkvGetNVocab Returns the number of tokens in the given model's vocabulary.
	// Useful for telling 20B_tokenizer models (n_vocab = 50277) apart from World models (n_vocab = 65536).
	RwkvGetNVocab()

	// RwkvGetNEmbedding Returns the number of elements in the given model's embedding.
	// Useful for reading individual fields of a model's hidden state.
	RwkvGetNEmbedding()

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
	RwkvQuantizeModelFile(in, out string, format RwkvQuantizedFormat) (bool, error)

	// RwkvGetSystemInfoString Returns system information string.
	RwkvGetSystemInfoString() string
}
