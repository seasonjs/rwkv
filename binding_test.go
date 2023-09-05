// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

package rwkv

import (
	"testing"
)

func assert(t *testing.T, con bool) {
	if !con {
		t.Error("fail with here, result is false")
	}
}

func assertNonNil(t *testing.T, obj any) {
	if obj == nil {
		t.Error("fail with here, obj is nil")
	}
}

func TestNewCRwkv(t *testing.T) {
	rwkv, err := NewCRwkv("./deps/windows/rwkv_avx_x64.dll")
	if err != nil {
		t.Error(err)
	}
	t.Log(rwkv)
	assertNonNil(t, rwkv)
}
