// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

//go:build windows && amd64

package rwkv

import (
	_ "embed"
	"golang.org/x/sys/cpu"
	"log"
) // Needed for go:embed

//go:embed deps/windows/rwkv_avx_x64.dll
var libRwkvAvx []byte

//go:embed deps/windows/rwkv_avx2_x64.dll
var libRwkvAvx2 []byte

//go:embed deps/windows/rwkv_avx512_x64.dll
var libRwkvAvx512 []byte

//go:embed deps/windows/rwkv_hipBLAS_GFX1100.dll
var libRwkvHipBLASGFX1100 []byte

var libName = "rwkv-*.dll"

var supportGpuTable = map[string][]byte{
	"AMD Radeon RX 7900 XTX": libRwkvHipBLASGFX1100,
	"AMD Radeon RX 7900 XT":  libRwkvHipBLASGFX1100,
}

func getDl(gpu bool) []byte {
	if gpu {
		gpuInfo, err := GetGPUInfo()
		if err != nil {
			log.Println(err)
		}
		if supportGpuTable[gpuInfo] != nil {
			return supportGpuTable[gpu]
		} else {
			log.Println("GPU not support, use CPU instead.")
		}
	}
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
