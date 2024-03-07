package main

import (
	"os"
	"testing"
)

func TestHandleNaive(t *testing.T) {
	expectedBytes, _ := os.ReadFile("../_data/measurements_1m_result.txt")
	expected := string(expectedBytes)
	actual := HandleNaive("../_data/measurements_1m.txt")
	if expected != actual {
		t.Errorf("Got:\n%s, expected:\n%s", actual, expected)
	}
}

func TestHandleBufio(t *testing.T) {
	expectedBytes, _ := os.ReadFile("../_data/measurements_1m_result.txt")
	expected := string(expectedBytes)
	actual := HandleBufio("../_data/measurements_1m.txt")
	if expected != actual {
		t.Errorf("Got:\n%s, expected:\n%s", actual, expected)
	}
}

func TestLengthToChunks(t *testing.T) {
	expected := []*Range{{start: 0, end: 3}, {start: 4, end: 7}, {start: 8, end: 11}, {start: 12, end: 15}}
	actual := LengthToChunks(16, 4)
	if len(expected) != len(actual) {
		t.Errorf("%v != %v", actual, expected)
	}
	for i := range len(expected) {
		if *actual[i] != *expected[i] {
			t.Errorf("(%d) %v != %v", i, *actual[i], *expected[i])
		}
	}
}

func TestHandleMmap(t *testing.T) {
	expectedBytes, _ := os.ReadFile("../_data/measurements_1m_result.txt")
	expected := string(expectedBytes)
	actual := HandleMmap("../_data/measurements_1m.txt")
	if expected != actual {
		t.Errorf("Got:\n%s, expected:\n%s", actual, expected)
	}
}
