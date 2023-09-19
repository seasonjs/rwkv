package rwkv

import (
	"bufio"
	"embed"
	"strconv"
	"strings"
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
func (t *Trie) Add(key, val string) {
	node := t.Root
	for _, ch := range key {
		char := string(ch)
		if node.to[char] == nil {
			node.to[char] = NewTrieNode()
		}
		node = node.to[char]
	}
	node.values[val] = true
}

// FindLongest finds the longest match in the trie for the given key
func (t *Trie) FindLongest(key string) string {
	node := t.Root
	pos, matchedKey := 0, ""
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
	return matchedKey
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
	for scanner.Scan() {
		line := scanner.Text()
		caps := strings.Split(line, " ")
		index, err := strconv.Atoi(caps[0])
		if err != nil {
			return nil, err
		}
		rest := caps[1]
		x, err := parseBytes(rest)
		if err != nil {
			return nil, err
		}
		token := string(x)
		wt.IndexToToken[index] = token
		wt.TokenToIndex[token] = index
		wt.Trie.Add(token, token)
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return wt, nil
}

// EncodeBytes encodes bytes to tokens
func (wt *WorldTokenizer) EncodeBytes(src []byte) []int {
	var tokens []int
	idx := 0
	for idx < len(src) {
		_idx := idx
		matchedKey := wt.Trie.FindLongest(string(src[idx:]))
		idx += len([]byte(matchedKey))
		if idx == _idx {
			panic("Invalid input")
		}
		tokens = append(tokens, wt.TokenToIndex[matchedKey])
	}
	return tokens
}

// DecodeBytes decodes tokens to bytes
func (wt *WorldTokenizer) DecodeBytes(tokens []int) []byte {
	result := []byte{}
	for _, token := range tokens {
		result = append(result, []byte(wt.IndexToToken[token])...)
	}
	return result
}

// Encode encodes a string to tokens
func (wt *WorldTokenizer) Encode(src string) ([]int, error) {
	return wt.EncodeBytes([]byte(src)), nil
}

// Decode decodes tokens to a string
func (wt *WorldTokenizer) Decode(tokens []int) string {
	return string(wt.DecodeBytes(tokens))
}

func parseBytes(s string) ([]byte, error) {
	s = strings.TrimSpace(s)

	if strings.HasPrefix(s, "'\\x") && strings.HasSuffix(s, "'") && len(s) > 2 {
		// handle \x...
		var bs []byte
		caps := strings.Split(s, "\\x")
		for _, cap := range caps {
			unquoted, err := strconv.Unquote("\\x" + cap)
			if err != nil {
				return nil, err
			}
			bs = append(bs, []byte(unquoted)...)
		}

		return bs, nil
	}
	if strings.HasPrefix(s, "b'") && strings.HasSuffix(s, "'") && len(s) > 3 {
		// handle b'...'
		return []byte(s[2 : len(s)-1]), nil
	}

	return []byte(strings.Trim(s, "'")), nil
}
