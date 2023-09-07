package rwkv

import (
	"github.com/sugarme/tokenizer"
	"testing"
)

func assertEncodeAndDecode(t *testing.T, tk *tokenizer.Tokenizer, input string) {
	in := tokenizer.NewSingleEncodeInput(tokenizer.NewInputSequence(input))
	encode, err := tk.Encode(in, false)
	if err != nil {
		t.Error(err)
	}

	decode := tk.Decode(encode.Ids, false)
	assert(t, input == decode)
}

func TestNormalTokenizer(t *testing.T) {
	tk, err := NormalTokenizer()
	if err != nil {
		t.Error(err)
	}
	seq1 := "hello world"
	t.Run("Test English", func(t *testing.T) {
		assertEncodeAndDecode(t, tk, seq1)
	})

	seq2 := "你好世界"
	t.Run("Test Chinese", func(t *testing.T) {
		assertEncodeAndDecode(t, tk, seq2)
	})

	seq3 := "こんにちは世界"
	t.Run("Test Japanese", func(t *testing.T) {
		assertEncodeAndDecode(t, tk, seq3)
	})
}
