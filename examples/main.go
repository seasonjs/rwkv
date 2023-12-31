package main

import (
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
		GpuEnable:     true,
	})

	if err != nil {
		print(err.Error())
		return
	}

	defer model.Close()

	err = model.LoadFromFile("./models/RWKV-5-World-0.4B-v2-20231113-ctx4096-F16.bin")
	if err != nil {
		print(err.Error())
		return
	}

	ctx, err := model.InitState()
	if err != nil {
		print(err.Error())
		return
	}
	out, err := ctx.Predict(
		`天元大陆上有五个国家，分别是北方的天金帝国，南方的华盛帝国，西方的落日帝国和东方的索域联邦，
而处于四大国中央，分别和四国接壤的一片面积不大呈六角形的土地就是天元大陆上最著名的神圣教廷。
四大王国中除了落日帝国和华盛帝国关系不佳以外，其他国家到是可以和平相处。每年，各个国家都要向教廷上交一定的“保护费”以作为教廷的开销。`)
	if err != nil {
		print(err.Error())
		return
	}
	print(out)
}
