// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

//go:build linux

package rwkv

import (
	_ "embed" // Needed for go:embed
)

//go:embed deps/linux/librwkv.so
var libRwkv []byte

var libName = "librwkv-*.so"

func getDl(gpu bool) []byte {
	if gpu {
		panic("Automatic loading of dynamic library failed, GPU Setting Not support linux Now. Push request is welcome.")
	}

	return libRwkv
}
