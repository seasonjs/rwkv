// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

//go:build windows && amd64

package rwkv

import (
	_ "embed"
	"golang.org/x/sys/cpu"
) // Needed for go:embed

//go:embed deps/windows/rwkv_avx_x64.dll
var libRwkvAvx []byte

//go:embed deps/windows/rwkv_avx2_x64.dll
var libRwkvAvx2 []byte

//go:embed deps/windows/rwkv_avx512_x64.dll
var libRwkvAvx512 []byte

//go:embed deps/windows/rwkv_hipBLAS.dll
var libRwkvHipLAS []byte

var libName = "rwkv-*.dll"

func getDl() []byte {
	if cpu.X86.HasAVX512 {
		return libRwkvAvx512
	}
	if cpu.X86.HasAVX2 {
		return libRwkvAvx2
	}
	if cpu.X86.HasAVX {
		return libRwkvAvx
	}
	//return libRwkvHipLAS
	panic("Automatic loading of dynamic library failed, please use `NewRwkvModel` method load manually. ")
	return nil
}
