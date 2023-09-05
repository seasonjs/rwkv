// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

//go:build darwin

package rwkv

import (
	"github.com/ebitengine/purego"
	"runtime"
)

var libraryPath = "deps/darwin/librwkv_x86.dylib"

func init() {
	if runtime.GOARCH == "arm64" {
		libraryPath = "deps/darwin/librwkv_arm64.dylib"
	}
}

func openLibrary(name string) (uintptr, error) {
	return purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
}
