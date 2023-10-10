// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

package rwkv

import (
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
	tk, err := NewWorldTokenizer()
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
	seq4 := "Привет, мир"
	t.Run("Test Russian", func(t *testing.T) {
		assertEncodeAndDecode(t, tk, seq4)
	})
	// rwkv_vocab_v20230424.txt not support Korean, such as `녕`
	//seq5 := "안녕 세상"
	//t.Run("Test Korean", func(t *testing.T) {
	//	assertEncodeAndDecode(t, tk, seq5)
	//})
	seq6 := "、]) -> <|endoftext|><|padding|>"
	t.Run("Test Special Characters", func(t *testing.T) {
		assertEncodeAndDecode(t, tk, seq6)
	})

	seq7 := "\n \n\n \t\t \t"
	t.Run("Test table character", func(t *testing.T) {
		assertEncodeAndDecode(t, tk, seq7)
	})

}

func TestEncodeUtf8(t *testing.T) {
	t.Run("Test new line to ASCII", func(t *testing.T) {
		str := string([]rune{
			'\\',
			'n',
			'\\',
			'n',
		})
		// Handle quoted strings with escape sequences.
		b := encodeUtf8(str)
		r := []rune(b)
		t.Log(r)
		assert(t, len(r) > 0)
	})
	t.Run("Test new Table to ASCII", func(t *testing.T) {
		str := string([]rune{
			'\\',
			't',
			'\\',
			't',
		})
		// Handle quoted strings with escape sequences.
		b := encodeUtf8(str)
		r := []rune(b)
		t.Log(r)
		assert(t, len(r) == 2)
	})
	t.Run("Test special char", func(t *testing.T) {
		str := string([]rune{
			'\x80',
			'\x81',
			'\x91',
			'\x92',
			'\x93',
			'\x94',
			'\x97',
		})
		// Handle quoted strings with escape sequences.
		b := encodeUtf8(str)
		r := []rune(b)
		assert(t, len(r) == 14)
		t.Log(r)
	})
}
