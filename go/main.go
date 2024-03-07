package main

import (
	"fmt"
	"log"
	"os"
	"runtime/pprof"
)

func main() {
	args := os.Args
	if len(args) > 1 && args[1] == "pprof" {
		log.Print("Profiling..")
		err := os.MkdirAll("profile", os.ModePerm)
		check(err)
		f, perr := os.Create("profile/cpu.pprof")
		if perr != nil {
			log.Fatal(perr)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}
	// fmt.Print(HandleNaive("../_data/measurements_1b.txt"))
	// fmt.Print(HandleBufio("../_data/measurements_1b.txt"))
	fmt.Print(HandleMmap("../_data/measurements_1b.txt"))
}
