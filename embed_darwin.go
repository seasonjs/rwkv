// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

//go:build darwin

package rwkv

import (
	_ "embed" // Needed for go:embed
	"runtime"
)

//go:embed deps/darwin/librwkv_x86.dylib
var libRwkvAmd64 []byte

//go:embed deps/darwin/librwkv_arm64.dylib
var libRwkvArm []byte

var libName = "librwkv-*.dylib"

func getDl(gpu bool) []byte {
	if runtime.GOARCH == "amd64" {
		return libRwkvAmd64
	}
	return libRwkvArm
}
