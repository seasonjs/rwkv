# rwkv

pure go for rwkv and support cross-platform.

[![Go Reference](https://pkg.go.dev/badge/github.com/seasonjs/rwkv.svg)](https://pkg.go.dev/github.com/seasonjs/rwkv)

rwkv.go is a wrapper around [rwkv.cpp](https://github.com/saharNooby/rwkv.cpp), which is an adaption of ggml.cpp.

## Installation

```bash
go get github.com/seasonjs/rwkv
```

## AutoModel Compatibility

See `deps` folder for dylib compatibility,or you can [build the library by yourself](https://github.com/saharNooby/rwkv.cpp#option-22-build-the-library-yourself), and push request is welcome.

`NewRwkvAutoModel` both gpu support `AMD` and `NVIDIA` on Windows.

`NewRwkvModel` need you to load the dynamic library manually, and the dynamic library is platform dependent.

Windows AMD GPU User may need check [rocm architecture](https://rocm.docs.amd.com/en/latest/release/windows_support.html#windows-supported-gpus) to get more information.

Windows NVIDIA GPU User may need check [cuda architecture](https://developer.nvidia.com/cuda-gpus) to get more information.

| platform | x32         | x64                     | arm         | AMD/ROCM        | NVIDIA/CUDA    |
|----------|-------------|-------------------------|-------------|-----------------|----------------|
| windows  | not support | support avx/avx2/avx512 | not support | rocm5.5 support | cuda12 support |
| linux    | not support | support                 | not support | not support     | not support    |
| darwin   | not support | support                 | support     | not support     | not support    |

## AutoModel Dynamic Libraries Disclaimer

#### The Source Of Dynamic Libraries
These dynamic libraries come from [rwkv.cpp release](https://github.com/saharNooby/rwkv.cpp/releases), The dynamic library version can be obtained by viewing [rwkv.version file](./deps/rwkv.version)
Anyone can check the consistency of the file by checksum the md5 of the file. 

#### The Security Of Dynamic Libraries
All I can say is that the creation of the dynamic library is public and does not contain any subjective malicious logic.
If you are worried about the security of the dynamic library during the use process, you can build it yourself.

**I and any author related to dynamic libraries do not assume any problems, responsibilities or legal liability during use.**

## Usage

You can find a complete example in [examples](./examples) folder.

Here is a simple example:

```go
package main

import (
	"fmt"
	"github.com/seasonjs/rwkv"
)

func main() {
	model, err := rwkv.NewRwkvAutoModel(rwkv.RwkvOptions{
		MaxTokens:     500,
		StopString:    "\n\n",
		Temperature:   0.8,
		TopP:          0.5,
		TokenizerType: rwkv.World, //or World
		PrintError:    true,
		CpuThreads:    10,
		GpuEnable:     false,
	})

	if err != nil {
		fmt.Print(err.Error())
		return
	}

	defer model.Close()

	err = model.LoadFromFile("./models/RWKV-5-World-0.4B-v2-20231113-ctx4096-F16.bin")
	if err != nil {
		fmt.Print(err.Error())
		return
	}
	prompt := `The following is a coherent verbose detailed conversation between a Chinese girl named Alice and her friend Bob.
Alice is very intelligent, creative and friendly.
Alice likes to tell Bob a lot about herself and her opinions.
Alice usually gives Bob kind, helpful and informative advices.

Bob: lhc
Alice: LHC是指大型强子对撞机（Large Hadron Collider），是世界最大最强的粒子加速器，由欧洲核子中心（CERN）在瑞士日内瓦地下建造。
LHC的原理是加速质子（氢离子）并让它们相撞，让科学家研究基本粒子和它们之间的相互作用，并在2012年证实了希格斯玻色子的存在。

Bob: 企鹅会飞吗
Alice: 企鹅是不会飞的。企鹅的翅膀短而扁平，更像是游泳时的一对桨。企鹅的身体结构和羽毛密度也更适合在水中游泳，而不是飞行。

`
	user := `Bob: 请介绍北京的旅游景点？
Alice: `

	ctx, err := model.InitState(prompt)

	if err != nil {
		print(err.Error())
		return
	}

	out, err := ctx.Predict(user)

	if err != nil {
		print(err.Error())
		return
	}

	print(out)
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
