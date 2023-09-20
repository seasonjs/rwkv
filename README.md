# rwkv

pure go for rwkv and support cross-platform

[![Go Reference](https://pkg.go.dev/badge/github.com/seasonjs/rwkv.svg)](https://pkg.go.dev/github.com/seasonjs/rwkv)

rwkv.go is a wrapper around [rwkv-cpp](https://github.com/saharNooby/rwkv.cpp), which is an adaption of ggml.cpp.

## Installation

```bash
go get github.com/seasonjs/rwkv
```

## Compatibility

See `deps` folder for dylib compatibility, push request is welcome.

| platform | x32         | x64                     | arm         |
|----------|-------------|-------------------------|-------------|
| windows  | not support | support avx2/avx512/avx | not support |
| linux    | not support | support                 | not support |
| darwin   | not support | support                 | support     |

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
		StopString:  "\n",
		Temperature: 0.8,
		TopP:        0.5,
		TokenizerType: rwkv.Normal, //or World 
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

	err = model.LoadFromFile("./data/rwkv-110M-Q5.bin", 2)
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
		StopString:  "\n",
		Temperature: 0.8,
		TopP:        0.5,
		TokenizerType: rwkv.Normal, //or World 
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

	err = model.LoadFromFile("./data/rwkv-110M-Q5.bin", 2)
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

## License

Copyright (c) seasonjs. All rights reserved.
Licensed under the MIT License. See License.txt in the project root for license information.