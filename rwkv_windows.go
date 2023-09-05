// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

//go:build windows

package rwkv

import (
	"golang.org/x/sys/cpu"
	"golang.org/x/sys/windows"
)

var libraryPath = "./deps/windows/rwkv_avx_x64.dll"

func init() {

	if cpu.X86.HasAVX {
		libraryPath = "./deps/windows/rwkv_avx_x64.dll"
	}
	if cpu.X86.HasAVX2 {
		libraryPath = "./deps/windows/rwkv_avx2_x64.dll"
	}
	if cpu.X86.HasAVX512 {
		libraryPath = "./deps/windows/rwkv_avx512_x64.dll"
	}
}

func openLibrary(name string) (uintptr, error) {
	handle, err := windows.LoadLibrary(name)
	return uintptr(handle), err
}
