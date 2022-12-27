package main

import (
	"fmt"
	"time"
)

func main() {

	start := time.Now()
	sum := 0

	for i := 0; i < 100000000; i++ {
		sum = sum + 1
	}

	end := time.Now()

	fmt.Println(end.Sub(start))
}
