package rwkv

import (
	"runtime"
	"strings"
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

func TestDllPath(t *testing.T) {
	t.Log(libraryPath)
	switch runtime.GOOS {
	case "windows":
		assert(t, strings.HasPrefix(libraryPath, "./deps/windows/rwkv_"))
	case "darwin":
		assert(t, strings.HasPrefix(libraryPath, "./deps/darwin/"))
	case "linux":
		assert(t, strings.HasPrefix(libraryPath, "./deps/linux/"))
	}

}

func TestNewCRwkv(t *testing.T) {
	rwkv, err := NewCRwkv()
	if err != nil {
		t.Error(err)
	}
	t.Log(rwkv)
	assertNonNil(t, rwkv)
}
