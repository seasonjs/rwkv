package rwkv

import (
	"fmt"
	"runtime"
	"testing"
)

func getLibrary() string {
	switch runtime.GOOS {
	case "darwin":
		return "./deps/darwin/librwkv_x86.dylib"
	case "linux":
		return "./deps/linux/librwkv.so"
	case "windows":
		return "./deps/windows/rwkv_avx2_x64.dll"
	default:
		panic(fmt.Errorf("GOOS=%s is not supported", runtime.GOOS))
	}
}
func TestRwkvModel(t *testing.T) {
	rwkv, err := NewRwkvModel(getLibrary(), RwkvOptions{
		maxTokens:   100,
		stopString:  "\n",
		temperature: 0.8,
		topP:        0.5,
	})

	if err != nil {
		t.Error(err)
	}

	defer func(rwkv *RwkvModel) {
		err := rwkv.Close()
		if err != nil {
			t.Error(err)
		}
	}(rwkv)

	rwkv.LoadFromFile("./data/tiny-rwkv-660K-FP16.bin", 2)
	t.Run("test predit", func(t *testing.T) {
		ctx, err := rwkv.InitState()
		if err != nil {
			t.Error(err)
		}
		out, err := ctx.Predict("hello ")
		if err != nil {
			t.Error(err)
		}
		assert(t, len(out) >= 0)
	})

	t.Run("test predict stream", func(t *testing.T) {
		ctx, err := rwkv.InitState()
		if err != nil {
			t.Error(err)
		}
		responseText := ""
		msg := make(chan string)
		ctx.PredictStream("hello ", msg)
		if err != nil {
			t.Error(err)
		}
		for value := range msg {
			responseText += value
		}
		assert(t, len(responseText) >= 0)
	})
}

func TestAutoLoad(t *testing.T) {
	rwkv, err := NewRwkvAutoModel(RwkvOptions{
		maxTokens:   100,
		stopString:  "\n",
		temperature: 0.8,
		topP:        0.5,
	})

	if err != nil {
		t.Error(err)
	}

	defer func(rwkv *RwkvModel) {
		err := rwkv.Close()
		if err != nil {
			t.Error(err)
		}
	}(rwkv)

	rwkv.LoadFromFile("./data/tiny-rwkv-660K-FP16.bin", 2)

	t.Run("test predict", func(t *testing.T) {
		ctx, err := rwkv.InitState()
		if err != nil {
			t.Error(err)
		}
		out, err := ctx.Predict("hello ")
		if err != nil {
			t.Error(err)
		}
		assert(t, len(out) >= 0)
	})

	t.Run("test predict stream", func(t *testing.T) {
		ctx, err := rwkv.InitState()
		if err != nil {
			t.Error(err)
		}
		responseText := ""
		msg := make(chan string)
		ctx.PredictStream("hello ", msg)
		if err != nil {
			t.Error(err)
		}
		for value := range msg {
			responseText += value
		}
		assert(t, len(responseText) >= 0)
	})

}
