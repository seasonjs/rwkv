# rwkv

pure go for rwkv and support cross-platform.

[![Go Reference](https://pkg.go.dev/badge/github.com/seasonjs/rwkv.svg)](https://pkg.go.dev/github.com/seasonjs/rwkv)

rwkv.go is a wrapper around [rwkv-cpp](https://github.com/saharNooby/rwkv.cpp), which is an adaption of ggml.cpp.

## Installation

```bash
go get github.com/seasonjs/rwkv
```

## AutoModel Compatibility

See `deps` folder for dylib compatibility, you can [build the library by yourself](https://github.com/saharNooby/rwkv.cpp#option-22-build-the-library-yourself), and push request is welcome.

So far `NewRwkvAutoModel` gpu only support `Windows ROCM GFX1100`.

If you want to use GPU, please make sure your GPU support `Windows ROCM GFX1100`.

Click here to see your `Windows ROCM`  [architecture](https://rocm.docs.amd.com/en/latest/release/windows_support.html#windows-supported-gpus).

| platform | x32         | x64                     | arm         | AMD/ROCM        | NVIDIA/CUDA |
|----------|-------------|-------------------------|-------------|-----------------|-------------|
| windows  | not support | support avx2/avx512/avx | not support | support GFX1100 | not support |
| linux    | not support | support                 | not support | not support     | not support |
| darwin   | not support | support                 | support     | not support     | not support |

## Usage

You don't need to download rwkv dynamic library.

```go
package main

import (
	"fmt"
	"github.com/seasonjs/rwkv"
)

func main() {
	model, err := rwkv.NewRwkvAutoModel(rwkv.RwkvOptions{
		MaxTokens:   100,
		StopString:  "\n\n",
		Temperature: 0.8,
		TopP:        0.5,
		TokenizerType: rwkv.Normal, //or World 
		CpuThreads:    10,
	})
	if err != nil {
		fmt.Print(err.Error())
	}
	defer func(rwkv *rwkv.RwkvModel) {
		err := model.Close()
		if err != nil {
			panic(err)
		}
	}(model)

	err = model.LoadFromFile("./data/rwkv-110M-Q5.bin")
	if err != nil {
		fmt.Print(err.Error())
	}

	// This context hold the logits and status, as well can int a new one.
	ctx, err := model.InitState()
	if err != nil {
		fmt.Print(err.Error())
	}
	out, err := ctx.Predict("hello ")
	if err != nil {
		fmt.Print(err.Error())
	}
	fmt.Print(out)

	// We can use `PredictStream` to generate response like `ChatGPT`

	ctx1, err := model.InitState()
	if err != nil {
		fmt.Print(err.Error())
	}
	responseText := ""
	msg := make(chan string)
	ctx1.PredictStream("hello ", msg)
	if err != nil {
		fmt.Print(err.Error())
	}
	for value := range msg {
		responseText += value
	}
	fmt.Print(responseText)
}

```
Now GPU is supported.you can use `NewRwkvAutoModel` to set `GpuEnable`. see `AutoModel Compatibility` about gpu support.

```go
package main

import (
	"fmt"
	"github.com/seasonjs/rwkv"
)

func main() {
	model, err := rwkv.NewRwkvAutoModel(rwkv.RwkvOptions{
		MaxTokens:     100,
		StopString:    "\n\n",
		Temperature:   0.8,
		TopP:          0.5,
		TokenizerType: World, //or World
		PrintError:    true,
		CpuThreads:    10,
		GpuEnable:     true,
		//GpuOffLoadLayers:      0, //default 0 means all layers will offload to gpu
	})

	if err != nil {
		panic(err)
	}

	defer model.Close()

	err = model.LoadFromFile("./models/RWKV-novel-4-World-7B-20230810-ctx128k-ggml-Q5_1.bin")
	if err != nil {
		fmt.Printf("error: %v", err)
		return
	}

	// NOTICE: This context hold the logits and status, as well can int a new one.
	ctx, err := model.InitState()
	if err != nil {
		panic(err)
	}
	out, err := ctx.Predict("hello ")
	if err != nil {
		fmt.Printf("error: %v", err)
	}
	fmt.Print(out)
}

```

If `NewRwkvAutoModel` can't automatic loading of dynamic library, please use `NewRwkvModel` method load manually.

```go
package main

import (
	"fmt"
	"github.com/seasonjs/rwkv"
	"runtime"
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

func main() {
	model, err := rwkv.NewRwkvModel(getLibrary(), rwkv.RwkvOptions{
		MaxTokens:   100,
		StopString:  "\n\n",
		Temperature: 0.8,
		TopP:        0.5,
		TokenizerType: rwkv.Normal, //or World 
		CpuThreads:    10,
	})
	if err != nil {
		fmt.Print(err.Error())
	}
	defer model.Close()

	err = model.LoadFromFile("./data/rwkv-110M-Q5.bin")
	if err != nil {
		fmt.Print(err.Error())
	}

	// This context hold the logits and status, as well can int a new one.
	ctx, err := model.InitState()
	if err != nil {
		fmt.Print(err.Error())
	}
	out, err := ctx.Predict("hello ")
	if err != nil {
		fmt.Print(err.Error())
	}
	fmt.Print(out)

	// We can use `PredictStream` to generate response like `ChatGPT`

	ctx1, err := model.InitState()
	if err != nil {
		fmt.Print(err.Error())
	}
	responseText := ""
	msg := make(chan string)
	ctx1.PredictStream("hello ", msg)
	if err != nil {
		fmt.Print(err.Error())
	}
	for value := range msg {
		responseText += value
	}
	fmt.Print(responseText)
}

```

### Chat With Bot
This library can be used to chat with bot.

```go

package main
import (
	"fmt"
	"github.com/seasonjs/rwkv"
)
func main()  {
	model, err := rwkv.NewRwkvAutoModel(rwkv.RwkvOptions{
		MaxTokens:     500,
		StopString:    "\n\n",
		Temperature:   0.8,
		TopP:          0.5,
		TokenizerType: World, //or World
		PrintError:    true,
		CpuThreads:    10,
		GpuEnable:     true,
	})

	if err != nil {
		fmt.Print(err.Error())
		return
	}

	defer model.Close()
	
	err = model.LoadFromFile("./models/RWKV-novel-4-World-7B-20230810-ctx128k-ggml-f16.bin")
	if err != nil {
		fmt.Print(err.Error())
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

	user := "Bob: 请介绍北京的旅游景点？" +
		"\n\n" +
		"Alice: "
	ctx, err := model.InitState(prompt)
	
	if err != nil {
		fmt.Print(err.Error())
	}
	
	out, err := ctx.Predict(user)
	
	if err != nil {
		fmt.Print(err.Error())
	}
	
	fmt.Print(out)
}
```

## Packaging

To ship a working program that includes this AI, you will need to include the following files:

* librwkv.dylib / librwkv.so / rwkv.dll (buildin)
* the model file
* the tokenizer file (buildin)

## Low level API

This package also provide low level Api which is same as [rwkv-cpp](https://github.com/saharNooby/rwkv.cpp).
See detail at [rwkv-doc](https://pkg.go.dev/github.com/seasonjs/rwkv).

## Thanks

* [rwkv-cpp](https://github.com/saharNooby/rwkv.cpp)
* [ggml.cpp](https://github.com/saharNooby/ggml.cpp)
* [purego](https://github.com/ebitengine/purego)
* [go-rwkv.cpp](https://github.com/donomii/go-rwkv.cpp)
* [sugarme/tokenizer](https://github.com/sugarme/tokenizer)

## Sponsor

Special thanks to [JetBrains support](https://jb.gg/OpenSourceSupport) for sponsoring.

![JetBrains Logo (Main) logo](https://resources.jetbrains.com/storage/products/company/brand/logos/jb_beam.svg)

## License

Copyright (c) seasonjs. All rights reserved.
Licensed under the MIT License. See License.txt in the project root for license information.
