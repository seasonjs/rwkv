//go:build linux

package rwkv

var libraryPath = "./deps/linux/librwkv.so"

func openLibrary(name string) (uintptr, error) {
	return purego.Dlopen(name, purego.RTLD_NOW|purego.RTLD_GLOBAL)
}
