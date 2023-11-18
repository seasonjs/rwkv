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
		CpuThreads:    2,
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

	err = rwkv.LoadFromFile("./data/rwkv-169M.bin")
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
		StopString:    "<|endoftext|>",
		Temperature:   0.8,
		TopP:          0.5,
		TokenizerType: World, //or World
		PrintError:    true,
		CpuThreads:    2,
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

	err = rwkv.LoadFromFile("./models/RWKV-novel-4-World-7B-20230810-ctx128k-ggml-Q5_1.bin")
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

func TestRwkvModel_QuantizeModelFile(t *testing.T) {
	rwkv, err := NewRwkvAutoModel(RwkvOptions{
		MaxTokens:     100,
		StopString:    "\n",
		Temperature:   0.8,
		TopP:          0.5,
		TokenizerType: World, //or World
		PrintError:    true,
	})
	if err != nil {
		t.Error(err)
		return
	}

	defer rwkv.Close()

	err = rwkv.QuantizeModelFile("./models/RWKV-novel-4-World-7B-20230810-ctx128k-ggml-f16.bin", "./models/RWKV-novel-4-World-7B-20230810-ctx128k-ggml-Q4_0.bin", Q4_0)
	if err != nil {
		t.Error(err)
		return
	}
}

func TestNewRwkvAutoModelGPU(t *testing.T) {
	rwkv, err := NewRwkvAutoModel(RwkvOptions{
		MaxTokens:     50,
		StopString:    "\\n\\n",
		Temperature:   0.8,
		TopP:          0.5,
		TokenizerType: World, //or World
		PrintError:    true,
		CpuThreads:    10,
		GpuEnable:     true,
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

	err = rwkv.LoadFromFile("./models/RWKV-novel-4-World-7B-20230810-ctx128k-ggml-f16.bin")
	if err != nil {
		t.Error(err)
		return
	}

	t.Run("test predict", func(t *testing.T) {
		ctx, err := rwkv.InitState()
		if err != nil {
			t.Error(err)
		}
		out, err := ctx.Predict("天元大陆上有五个国家，分别是北方的天金帝国，南方的华盛帝国，西方的落日帝国和东方的索域联邦，而处于四大国中央，分别和四国接壤的一片面积不大呈六角形的土地就是天元大陆上最著名的神圣教廷。" +
			"四大王国中除了落日帝国和华盛帝国关系不佳以外，其他国家到是可以和平相处。" +
			"每年，各个国家都要向教廷上交一定的“保护费”以作为教廷的开销。")
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

func TestChat(t *testing.T) {
	rwkv, err := NewRwkvAutoModel(RwkvOptions{
		MaxTokens:     500,
		StopString:    "\n\n",
		Temperature:   0.8,
		TopP:          0.5,
		TokenizerType: World, //or World
		PrintError:    true,
		CpuThreads:    2,
		GpuEnable:     true,
	})

	if err != nil {
		t.Error(err)
		return
	}

	defer rwkv.Close()
	err = rwkv.LoadFromFile("./models/RWKV-5-World-3B-v2-f16.bin")
	if err != nil {
		t.Error(err)
		return
	}
	prompt := "The following is a coherent verbose detailed conversation between a Chinese girl named Alice and her friend Bob." +
		" Alice is very intelligent, creative and friendly." +
		" Alice likes to tell Bob a lot about herself and her opinions." +
		" Alice usually gives Bob kind, helpful and informative advices." +
		"\n\n" +
		"Bob: lhc" +
		"\n\n" +
		"Alice: LHC是指大型强子对撞机（Large Hadron Collider），是世界最大最强的粒子加速器，由欧洲核子中心（CERN）在瑞士日内瓦地下建造。" +
		"LHC的原理是加速质子（氢离子）并让它们相撞，让科学家研究基本粒子和它们之间的相互作用，并在2012年证实了希格斯玻色子的存在。" +
		"\n\n" +
		"Bob: 企鹅会飞吗" +
		"\n\n" +
		"Alice: 企鹅是不会飞的。企鹅的翅膀短而扁平，更像是游泳时的一对桨。" +
		"企鹅的身体结构和羽毛密度也更适合在水中游泳，而不是飞行。" +
		"\n\n"

	user := "Question: 请介绍北京的旅游景点？" +
		"\n\n" +
		"Answer: "
	t.Run("test chat with Chinese", func(t *testing.T) {
		ctx, err := rwkv.InitState(prompt)
		if err != nil {
			t.Error(err)
		}
		out, err := ctx.Predict(user)
		if err != nil {
			t.Error(err.Error())
		}
		t.Log(out)
		assert(t, len(out) >= 0)
	})

}

func TestRwkvState_GetEmbedding(t *testing.T) {
	rwkv, err := NewRwkvAutoModel(RwkvOptions{
		MaxTokens:     500,
		TokenizerType: Normal, //or World
		PrintError:    true,
		CpuThreads:    2,
		GpuEnable:     true,
	})
	defer rwkv.Close()

	err = rwkv.LoadFromFile("./models/RWKV-4b-Pile-171M-20230202-7922-f16.bin")
	if err != nil {
		t.Error(err)
		return
	}
	ctx, err := rwkv.InitState()
	if err != nil {
		t.Error(err)
		return
	}
	nb := ctx.rwkvModel.cRwkv.RwkvGetNEmbedding(ctx.rwkvModel.ctx)
	t.Log(nb)
	nl := ctx.rwkvModel.cRwkv.RwkvGetNLayer(ctx.rwkvModel.ctx)

	t.Run("hidden state", func(t *testing.T) {
		embedding, err := ctx.GetEmbedding("hello word", false)
		if err != nil {
			t.Error(err)
			return
		}
		t.Log(embedding)
		t.Log(len(embedding))
		t.Log(nl)
		assert(t, len(embedding) == int(nb*5*nl))
	})

	t.Run("distill hidden state ", func(t *testing.T) {
		embedding, err := ctx.GetEmbedding("hello word", true)
		if err != nil {
			t.Error(err)
			return
		}
		t.Log(embedding)
		assert(t, len(embedding) == int(nb))
	})

}
