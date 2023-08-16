package rwkv

import (
	"strings"
	"testing"
)

func assertEncodeAndDecode(t *testing.T, tk Tokenizer, input string) {
	encode, err := tk.Encode(input)
	if err != nil {
		t.Error(err)
	}
	decode := tk.Decode(encode)
	assert(t, input == decode)
}

func TestNormalTokenizer(t *testing.T) {
	tk, err := NewNormalTokenizer()
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

func TestWorldTokenizer(t *testing.T) {
	tk, err := NewWordLevelTokenizer()
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

func TestBytes(t *testing.T) {

	a := []byte("\x00")
	t.Log(len(a))

	b := "'\x00'"
	t.Log("b:", len(b))
	d := strings.Trim(b, "'")
	c := []byte(b)
	t.Log("c", len(c))
	t.Log("d", len(d))
	t.Log("d bytes", len([]byte(d)))

}
