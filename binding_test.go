package rwkv

import (
	"reflect"
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
	}
}

func TestNewCRwkv(t *testing.T) {
	rwkv, err := NewCRwkv()
	if err != nil {
		t.Error(err)
	}
	t.Log(rwkv)
	// 获取结构体类型
	reflectType := reflect.TypeOf(rwkv)
	for i := 0; i < reflectType.NumField(); i++ {
		//field := reflectType.Field(i)
		value := reflect.ValueOf(rwkv).Field(i)
		assertNonNil(t, value.Interface())
	}
}
