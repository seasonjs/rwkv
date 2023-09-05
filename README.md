# rwkv

pure go for rwkv and support cross-platform

[![Go Reference](https://pkg.go.dev/badge/github.com/seasonjs/rwkv.svg)](https://pkg.go.dev/github.com/seasonjs/rwkv.cpp)

rwkv.go is a wrapper around [rwkv-cpp](https://github.com/saharNooby/rwkv.cpp), which is an adaption of ggml.cpp.

## Installation

```bash
go get github.com/seasonjs/rwkv
```

You may need to download dependencies (which called dynamic library), you can got it in release page, or get from `deps` folder.


## Packaging

To ship a working program that includes this AI, you will need to include the following files:

* librwkv.dylib / librwkv.so / rwkv.dll
* the model file
* the tokenizer file

## Thanks

* [rwkv-cpp](https://github.com/saharNooby/rwkv.cpp)
* [ggml.cpp](https://github.com/saharNooby/ggml.cpp)
* [purego](https://github.com/ebitengine/purego)
* [go-rwkv.cpp](https://github.com/donomii/go-rwkv.cpp)

## License

Copyright (c) seasonjs. All rights reserved.
Licensed under the MIT License. See License.txt in the project root for license information.