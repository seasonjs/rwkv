// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

package rwkv

import "testing"

func TestGetGPUInfo(t *testing.T) {
	info, err := getGPUInfo()
	if err != nil {
		return
	}
	t.Log(info)
}
