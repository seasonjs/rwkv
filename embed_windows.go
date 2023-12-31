// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

//go:build windows && amd64

package rwkv

import (
	_ "embed"
	"errors"
	"golang.org/x/sys/cpu"
	"log"
	"os/exec"
	"strings"
) // Needed for go:embed

//go:embed deps/windows/rwkv_avx.dll
var libRwkvAvx []byte

//go:embed deps/windows/rwkv_avx2.dll
var libRwkvAvx2 []byte

//go:embed deps/windows/rwkv_avx512.dll
var libRwkvAvx512 []byte

//go:embed deps/windows/rwkv_rocm5.5.dll
var libRwkvRocm []byte

//go:embed deps/windows/rwkv_cuda12.dll
var rwkvCuda []byte

var libName = "rwkv-*.dll"

func getDl(gpu bool) []byte {
	if gpu {
		info, err := getGPUInfo()
		if err != nil {
			log.Println(err)
		}
		log.Print("get gpu info: ", info["Name"])

		if strings.Contains(info["Name"], "AMD") {
			log.Println("Use GPU ROCM5.5 instead.")
			return libRwkvRocm
		}

		if strings.Contains(info["Name"], "NVIDIA") {
			log.Println("Use GPU CUDA12 instead.")
			return rwkvCuda
		}

		log.Println("GPU not support, use CPU instead.")
	}

	if cpu.X86.HasAVX512 {
		log.Println("Use CPU AVX512 instead.")
		return libRwkvAvx512
	}

	if cpu.X86.HasAVX2 {
		log.Println("Use CPU AVX2 instead.")
		return libRwkvAvx2
	}

	if cpu.X86.HasAVX {
		log.Println("Use CPU AVX instead.")
		return libRwkvAvx
	}

	panic("Automatic loading of dynamic library failed, please use `NewRwkvModel` method load manually. ")
	return nil
}

func runPowerShellCommand(command string) (string, error) {
	cmd := exec.Command("powershell", "-Command", command)

	// execute the command and get the output
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}

	return string(output), nil
}

func getGPUInfo() (map[string]string, error) {
	psCommand := "Get-WmiObject Win32_VideoController"
	output, err := runPowerShellCommand(psCommand)
	if err != nil {
		return nil, err
	}
	infos := strings.Split(output, "\r\n")
	result := make(map[string]string, len(infos))
	for _, line := range infos {
		parts := strings.SplitN(line, ":", 2)
		if len(parts) == 2 {
			key := strings.TrimSpace(parts[0])
			if strings.Contains(key, "__") || strings.Contains(key, "/") {
				continue
			}
			value := strings.TrimSpace(parts[1])
			if len(value) == 0 {
				continue
			}
			if len(value) == 0 {
				continue
			}
			result[key] = value
		}
	}
	if len(result["Name"]) > 0 {
		return result, nil
	}
	return nil, errors.New("no gpu found")
}
