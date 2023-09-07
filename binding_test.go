// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

package rwkv

import (
	"testing"
)

func assert(t *testing.T, con bool, message ...string) {
	if !con {
		if len(message) == 0 {
			t.Error("fail with here, result is false")
		} else {
			t.Error(message)
		}
	}
}

func assertNonNil(t *testing.T, obj any, message ...string) {
	if obj == nil {
		if len(message) == 0 {
			t.Error("fail with here, obj is nil")
		} else {
			t.Error(message)
		}
	}
}

func TestNewCRwkv(t *testing.T) {
	rwkv, err := NewCRwkv(getLibrary())
	if err != nil {
		t.Error(err)
	}
	t.Log(rwkv)
	assertNonNil(t, rwkv)
}
