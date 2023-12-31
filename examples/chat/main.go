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
