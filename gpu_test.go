package rwkv

import "testing"

func TestGetGPUInfo(t *testing.T) {
	info, err := GetGPUInfo()
	if err != nil {
		return
	}
	t.Log(info)
}
