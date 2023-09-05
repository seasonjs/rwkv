// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

package rwkv

type RwkvErrors uint32

// Represents an error encountered during a function call.
// These are flags, so an actual value might contain multiple errors.
const (
	RwkvErrorNone RwkvErrors = iota
	RwkvErrorAlloc
	RwkvErrorFileOpen
	RwkvErrorFileStat
	RwkvErrorFileRead
	RwkvErrorFileWrite
	RwkvErrorFileMagic
	RwkvErrorFileVersion
	RwkvErrorDataType
	RwkvErrorUnsupported
	RwkvErrorShape
	RwkvErrorDimension
	RwkvErrorKey
	RwkvErrorData
	RwkvErrorParamMissing
)
const (
	RwkvErrorArgs        RwkvErrors = 1 << 8
	RwkvErrorFile                   = 2 << 8
	RwkvErrorModel                  = 3 << 8
	RwkvErrorModelParams            = 4 << 8
	RwkvErrorGraph                  = 5 << 8
	RwkvErrorCtx                    = 6 << 8
)

var rwkvErrorMap = map[RwkvErrors]string{
	RwkvErrorNone:         "RWKV_ERROR_NONE",
	RwkvErrorAlloc:        "RWKV_ERROR_ALLOC",
	RwkvErrorFileOpen:     "RWKV_ERROR_FILE_OPEN",
	RwkvErrorFileStat:     "RWKV_ERROR_FILE_STAT",
	RwkvErrorFileRead:     "RWKV_ERROR_FILE_READ",
	RwkvErrorFileWrite:    "RWKV_ERROR_FILE_WRITE",
	RwkvErrorFileMagic:    "RWKV_ERROR_FILE_MAGIC",
	RwkvErrorFileVersion:  "RWKV_ERROR_FILE_VERSION",
	RwkvErrorDataType:     "RWKV_ERROR_DATA_TYPE",
	RwkvErrorUnsupported:  "RWKV_ERROR_UNSUPPORTED",
	RwkvErrorShape:        "RWKV_ERROR_SHAPE",
	RwkvErrorDimension:    "RWKV_ERROR_DIMENSION",
	RwkvErrorKey:          "RWKV_ERROR_KEY",
	RwkvErrorData:         "RWKV_ERROR_DATA",
	RwkvErrorParamMissing: "RWKV_ERROR_PARAM_MISSING",
	RwkvErrorArgs:         "RWKV_ERROR_ARGS",
	RwkvErrorFile:         "RWKV_ERROR_FILE",
	RwkvErrorModel:        "RWKV_ERROR_MODEL",
	RwkvErrorModelParams:  "RWKV_ERROR_MODEL_PARAMS",
	RwkvErrorGraph:        "RWKV_ERROR_GRAPH",
	RwkvErrorCtx:          "RWKV_ERROR_CTX",
}

func (err RwkvErrors) Error() string {
	return rwkvErrorMap[err]
}
