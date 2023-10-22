// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

//go:build linux

package rwkv

import (
	_ "embed" // Needed for go:embed
	"runtime"
)

//go:embed deps/linux/librwkv.so
var libRwkv []byte

var libName = "librwkv-*.so"

func getDl(gpu bool) []byte {
	if gpu {
		_, _ = GetGPUInfo()
	}
	if runtime.GOARCH == "amd64" {
		return libRwkv
	}

	panic("Automatic loading of dynamic library failed, please use `NewRwkvModel` method load manually. ")
	return nil
}
