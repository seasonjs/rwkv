// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

//go:build darwin

package rwkv

import (
	_ "embed" // Needed for go:embed
)

//go:embed deps/darwin/librwkv.dylib
var libRwkv []byte

var libName = "librwkv-*.dylib"

func getDl(gpu bool) []byte {
	if gpu {
		panic("Automatic loading of dynamic library failed, GPU Setting Not support darwin Now. Push request is welcome.")
	}

	return libRwkv
}
