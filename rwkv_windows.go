// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

//go:build windows

package rwkv

import (
	"golang.org/x/sys/cpu"
	"golang.org/x/sys/windows"
)

var path = "./deps/windows/rwkv.dll"

func Init() {
	if cpu.X86.HasAVX {
		path = "./deps/windows/rwkv_avx.dll"
	}
	if cpu.X86.HasAVX2 {
		path = "./deps/windows/rwkv_avx2.dll"
	}
	if cpu.X86.HasAVX512 {
		path = "./deps/windows/rwkv_avx512.dll"
	}
}

func openLibrary(name string) (uintptr, error) {
	handle, err := windows.LoadLibrary(name)
	return uintptr(handle), err
}
