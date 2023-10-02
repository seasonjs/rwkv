// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

//go:build windows && amd64

package rwkv

import (
	"errors"
	"os/exec"
	"regexp"
	"strings"
)

func runPowerShellCommand(command string) (string, error) {
	cmd := exec.Command("powershell", "-Command", command)

	// execute the command and get the output
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}

	return string(output), nil
}

func GetGPUInfo() (string, error) {
	psCommand := "Get-WmiObject Win32_VideoController"

	output, err := runPowerShellCommand(psCommand)
	if err != nil {
		return "", err
	}
	infos := strings.Split(output, "\r\n")
	re := regexp.MustCompile(`^Name\s+:\s+(.+)`)
	for _, info := range infos {
		match := re.FindStringSubmatch(info)
		if len(match) >= 2 {
			// get the first match Name
			gpuName := match[1]
			return gpuName, nil
		}
	}

	return "", errors.New("no gpu found")
}
