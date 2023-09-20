package rwkv

import (
	"fmt"
	"runtime"
	"testing"
)

func getLibrary() string {
	switch runtime.GOOS {
	case "darwin":
		return "./deps/darwin/librwkv_arm64.dylib"
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
		MaxTokens:     100,
		StopString:    "\n",
		Temperature:   0.8,
		TopP:          0.5,
		TokenizerType: Normal, //or World
	})

	if err != nil {
		t.Error(err)
		return
	}

	defer func(rwkv *RwkvModel) {
		err := rwkv.Close()
		if err != nil {
			t.Error(err)
		}
	}(rwkv)

	err = rwkv.LoadFromFile("./data/rwkv-169M.bin", 2)
	if err != nil {
		t.Error(err)
		return
	}
	t.Run("test predit", func(t *testing.T) {
		ctx, err := rwkv.InitState()
		if err != nil {
			t.Error(err)
		}
		out, err := ctx.Predict("hello world")
		if err != nil {
			t.Error(err.Error())
		}
		t.Log(out)
		assert(t, len(out) >= 0)
	})

	t.Run("test predict stream", func(t *testing.T) {
		ctx, err := rwkv.InitState()
		if err != nil {
			t.Error(err)
		}
		responseText := ""
		msg := make(chan string)
		ctx.PredictStream("hello world", msg)
		for value := range msg {
			responseText += value
		}
		t.Log(responseText)
		assert(t, len(responseText) >= 0)
	})
}

func TestAutoLoad(t *testing.T) {
	rwkv, err := NewRwkvAutoModel(RwkvOptions{
		MaxTokens:     100,
		StopString:    "\n",
		Temperature:   0.8,
		TopP:          0.5,
		TokenizerType: Normal, //or World
	})

	if err != nil {
		t.Error(err)
		return
	}

	defer func(rwkv *RwkvModel) {
		err := rwkv.Close()
		if err != nil {
			t.Error(err)
		}
	}(rwkv)

	err = rwkv.LoadFromFile("./data/rwkv-169M.bin", 2)
	if err != nil {
		t.Error(err)
		return
	}

	t.Run("test predict", func(t *testing.T) {
		ctx, err := rwkv.InitState()
		if err != nil {
			t.Error(err)
		}
		out, err := ctx.Predict("hello ")
		if err != nil {
			t.Error(err.Error())
		}
		t.Log(out)
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
		t.Log(responseText)
		assert(t, len(responseText) >= 0)
	})

}
