// Copyright (c) seasonjs. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

package rwkv

import (
	"bufio"
	"embed"
	"errors"
	"fmt"
	"log"
	"strconv"
	"strings"
	"unicode/utf8"
)

//go:embed rwkv_vocab_v20230424.txt
var worldTokenizerFS embed.FS

// TrieNode represents a node in the trie
type TrieNode struct {
	to     map[string]*TrieNode
	values map[string]bool
}

// Trie represents the trie data structure
type Trie struct {
	Root *TrieNode
}

// NewTrieNode initializes a new trie node
func NewTrieNode() *TrieNode {
	return &TrieNode{
		to:     make(map[string]*TrieNode),
		values: make(map[string]bool),
	}
}

// Add inserts a key into the trie
func (t *Trie) Add(val string) {
	node := t.Root
	for _, ch := range []rune(val) {
		char := string(ch)
		if node.to[char] == nil {
			node.to[char] = NewTrieNode()
		}
		node = node.to[char]
	}
	node.values[val] = true
}

// FindLongest finds the longest match in the trie for the given key
func (t *Trie) FindLongest(key []rune) string {
	node := t.Root
	var matchedKey []rune
	pos := 0
	for i, ch := range key {
		char := string(ch)
		if node.to[char] == nil {
			break
		}
		node = node.to[char]
		if len(node.values) > 0 {
			pos = i + 1
			matchedKey = key[:pos]
		}
	}
	return string(matchedKey)
}

// WorldTokenizer represents a tokenizer for encoding and decoding bytes to tokens
type WorldTokenizer struct {
	IndexToToken map[int]string
	TokenToIndex map[string]int
	Trie         *Trie
}

// NewWorldTokenizer initializes a new world tokenizer
func NewWorldTokenizer() (*WorldTokenizer, error) {
	f, err := worldTokenizerFS.Open("rwkv_vocab_v20230424.txt")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	wt := &WorldTokenizer{
		IndexToToken: make(map[int]string),
		TokenToIndex: make(map[string]int),
		Trie:         &Trie{Root: NewTrieNode()},
	}

	scanner := bufio.NewScanner(f)
	nonStandardToken := 0
	for scanner.Scan() {
		line := scanner.Text()
		fIndex := strings.Index(line, " ")
		lIndex := strings.LastIndex(line, " ")
		index, err := strconv.Atoi(line[:fIndex])
		if err != nil {
			return nil, err
		}
		expectLen, err := strconv.Atoi(line[lIndex+1:])
		if err != nil {
			return nil, err
		}
		rest := line[fIndex+1 : lIndex]
		token, err := parseBytes(rest)
		if err != nil {
			return nil, err
		}
		if expectLen != len(token) {
			if strings.ContainsRune(token, utf8.RuneError) {
				nonStandardToken++
			}
			log.Print("parse vocabulary to bytes fail,token index is:", index,
				" token is: ", token,
				", except length is: ", expectLen,
				", parse length is: ", len(token),
				", peek token bytes: ", []rune(token))
		}
		wt.IndexToToken[index] = token
		wt.TokenToIndex[token] = index
		wt.Trie.Add(token)
	}

	if nonStandardToken > 0 {
		log.Print("rwkv_vocab_v20230424.txt contains non-standard utf-8 total: ",
			nonStandardToken,
			". they will not affect your normal use")
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return wt, nil
}

// EncodeBytes encodes bytes to tokens
func (wt *WorldTokenizer) EncodeBytes(src []rune) ([]int, error) {
	var tokens []int
	idx := 0
	for idx < len(src) {
		matchedKey := wt.Trie.FindLongest(src[idx:])
		if len(matchedKey) <= 0 {
			return nil, fmt.Errorf("can't encode current language: %s", string(src[idx:]))
		}
		idx += len([]rune(matchedKey))
		tokens = append(tokens, wt.TokenToIndex[matchedKey])
	}
	return tokens, nil
}

// DecodeBytes decodes tokens to bytes
func (wt *WorldTokenizer) DecodeBytes(tokens []int) []rune {
	var result []rune
	for _, token := range tokens {
		result = append(result, []rune(wt.IndexToToken[token])...)
	}
	return result
}

// Encode encodes a string to tokens
func (wt *WorldTokenizer) Encode(src string) ([]int, error) {
	return wt.EncodeBytes([]rune(src))
}

// Decode decodes tokens to a string
func (wt *WorldTokenizer) Decode(tokens []int) string {
	return string(wt.DecodeBytes(tokens))
}

func parseBytes(s string) (string, error) {
	if strings.HasPrefix(s, "b'") && strings.HasSuffix(s, "'") && len(s) > 3 {
		// handle b'...'
		s = s[1:]
	}
	if len(s) <= 0 {
		return "", errors.New("rwkv_vocab_v20230424.txt vocab list broke, got vocab length equal zero")
	}
	// handle ''
	if strings.HasPrefix(s, "'") && strings.HasSuffix(s, "'") {
		s = s[1 : len(s)-1]
		b := encodeUtf8(s)
		b = ignoreBackslashes(b, false)
		return b, nil
	}
	// handle ""
	if strings.HasPrefix(s, "\"") && strings.HasSuffix(s, "\"") {
		b := ignoreBackslashes(s[1:len(s)-1], true)
		return b, nil
	}
	//
	return s, nil
}

// we should encodeUtf8 by python style.
func encodeUtf8(in string) string {
	var backslash byte = '\\'
	// Handle quoted strings with escape sequences.
	var buf []byte
	in0 := in
	buf = make([]byte, 0, (len(in)+1)*3/2) // try to avoid more allocations
	for len(in) > 0 {
		r, multibyte, rem, err := strconv.UnquoteChar(in, backslash)
		if err != nil {
			return in0
		}
		in = rem

		// Append the character if unescaping the input.
		if r < utf8.RuneSelf || !multibyte {
			buf = append(buf, byte(r))
		} else {
			var arr [utf8.UTFMax]byte
			n := utf8.EncodeRune(arr[:], r)
			buf = append(buf, arr[:n]...)
		}
	}
	return string(buf)
}

// we should ignoreBackslashes by python style.
func ignoreBackslashes(in string, doubleQuotes bool) string {
	var backslash byte = '\\'
	var quoteChar byte = '\''
	var buf []byte
	buf = make([]byte, 0, (len(in)+1)*3/2)
	for index, ch := range []byte(in) {
		if ch == backslash && len(in) > index+1 {
			if doubleQuotes {
				if in[index+1] != backslash {
					buf = append(buf, ch)
				}
			} else {
				if in[index+1] != quoteChar {
					buf = append(buf, ch)
				}
			}

		} else {
			buf = append(buf, ch)
		}
	}
	return string(buf)
}
