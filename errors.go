// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

package rwkv

type RwkvErrors uint32

// Represents an error encountered during a function call.
// These are flags, so an actual value might contain multiple errors.
const (
	RwkvErrorNone RwkvErrors = iota
	RWKV_ERROR_ALLOC
	RWKV_ERROR_FILE_OPEN
	RWKV_ERROR_FILE_STAT
	RWKV_ERROR_FILE_READ
	RWKV_ERROR_FILE_WRITE
	RWKV_ERROR_FILE_MAGIC
	RWKV_ERROR_FILE_VERSION
	RWKV_ERROR_DATA_TYPE
	RWKV_ERROR_UNSUPPORTED
	RWKV_ERROR_SHAPE
	RWKV_ERROR_DIMENSION
	RWKV_ERROR_KEY
	RWKV_ERROR_DATA
	RWKV_ERROR_PARAM_MISSING
)
