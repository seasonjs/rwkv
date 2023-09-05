package deps

import "embed"

//go:embed deps/*
var content embed.FS
