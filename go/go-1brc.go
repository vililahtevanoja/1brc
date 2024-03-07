package main

import (
	"bufio"
	"bytes"
	"fmt"
	"log"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/exp/mmap"
)

type MeasurementStats struct {
	min   float64
	max   float64
	sum   float64
	count int32
}

func addStat(stats *MeasurementStats, measurement float64) {
	if measurement < stats.min {
		stats.min = measurement
	}
	if measurement > stats.max {
		stats.max = measurement
	}
	stats.sum += measurement
	stats.count += 1
}

func mergeStats(stats *MeasurementStats, other *MeasurementStats) {
	if other.min < stats.min {
		stats.min = other.min
	}
	if other.max > stats.max {
		stats.max = other.max
	}
	stats.sum += other.sum
	stats.count += other.count
}

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func HandleNaive(file string) string {
	start := time.Now()
	cmp := time.Now()
	m := make(map[string]*MeasurementStats)
	data, err := os.ReadFile(file)
	check(err)
	log.Printf("Read file (%s)", time.Since(cmp))
	cmp = time.Now()

	dataStr := string(data)
	log.Printf("Prepped data (%s)", time.Since(cmp))
	cmp = time.Now()
	for _, line := range strings.Split(strings.TrimSuffix(dataStr, "\n"), "\n") {
		city, value, _ := strings.Cut(line, ";")
		valueFloat, err := strconv.ParseFloat(value, 64)
		check(err)
		stats, found := m[city]
		if found {
			addStat(stats, valueFloat)
		} else {
			m[city] = &MeasurementStats{min: valueFloat, max: valueFloat, sum: valueFloat, count: 1}
		}
	}
	log.Printf("Went through data (%s)", time.Since(cmp))
	cmp = time.Now()

	results := make([]string, 0)

	for k, v := range m {
		results = append(results, fmt.Sprintf("%s=%.1f/%.1f/%.1f", k, v.min, (v.sum/float64(v.count)), v.max))
	}
	log.Printf("Crunched results (%s)", time.Since(cmp))
	cmp = time.Now()
	slices.Sort(results)
	log.Printf("Sorted results (%s)", time.Since(cmp))
	log.Printf("Total duration: %s", time.Since(start))
	return fmt.Sprintf("{%s}", strings.Join(results, ", "))
}

func HandleBufio(file string) string {
	start := time.Now()
	cmp := time.Now()
	m := make(map[string]*MeasurementStats)
	f, err := os.Open(file)
	check(err)
	defer f.Close()
	buf := make([]byte, 32*1024)
	scanner := bufio.NewScanner(f)
	scanner.Buffer(buf, len(buf))
	log.Printf("Read file (%s)", time.Since(cmp))
	cmp = time.Now()
	for scanner.Scan() {
		text := scanner.Text()
		if text == "" {
			continue
		}
		city, value, _ := strings.Cut(text, ";")
		valueFloat, err := strconv.ParseFloat(value, 64)
		check(err)
		stats, found := m[city]
		if found {
			addStat(stats, valueFloat)
		} else {
			m[city] = &MeasurementStats{min: valueFloat, max: valueFloat, sum: valueFloat, count: 1}
		}
	}
	log.Printf("Went through data (%s)", time.Since(cmp))
	cmp = time.Now()

	results := make([]string, 0)

	for k, v := range m {
		results = append(results, fmt.Sprintf("%s=%.1f/%.1f/%.1f", k, v.min, (v.sum/float64(v.count)), v.max))
	}
	log.Printf("Crunched results (%s)", time.Since(cmp))
	cmp = time.Now()
	slices.Sort(results)
	log.Printf("Sorted results (%s)", time.Since(cmp))
	log.Printf("Total duration: %s", time.Since(start))
	return fmt.Sprintf("{%s}", strings.Join(results, ", "))
}

type Range struct {
	start int64
	end   int64
}

func LengthToChunks(length int, chunkCount int) []*Range {
	chunkSize := length / chunkCount
	chunks := make([]*Range, 0)
	for i := range chunkCount {
		chunks = append(chunks, &Range{start: int64(i * chunkSize), end: int64((i * chunkSize) + chunkSize - 1)})
	}
	log.Printf("chunks: %v", chunks)
	for i := range len(chunks[1:]) {
		i = i + 1
		if chunks[i-1].end != chunks[i].start-1 {
			log.Panicf("%v != %v", *chunks[i-1], *chunks[i])
		}
	}
	chunks[len(chunks)-1].end = int64(length - 1)
	return chunks
}

// fn parse_chunk(mmap: &Mmap, range: (usize, usize)) -> FxHashMap<Vec<u8>, MeasurementStatsInt> {
//     let (prelim_start, prelim_end) = range;
//     let mut start = prelim_start;
//     while *mmap.get(start).unwrap() != b'\n' {
//         start += 1;
//     }
//     let mut end = prelim_end;
//     while *mmap.get(end).unwrap() != b'\n' {
//         end += 1;
//     }
//     let mut chunk = &mmap[start + 1..end];
//     return parse(&mut chunk);
// }

func AdjustChunks(reader *mmap.ReaderAt, chunks []*Range) {
	buf := make([]byte, 1)
	for _, chunk := range chunks {
		_, err := reader.ReadAt(buf, chunk.start)
		check(err)
		newStart := chunk.start
		for buf[0] != '\n' {
			newStart += 1
			_, err = reader.ReadAt(buf, newStart)
			check(err)
		}
		reader.ReadAt(buf, chunk.end)
		check(err)
		newEnd := chunk.end
		for buf[0] != '\n' {
			newEnd += 1
			_, err = reader.ReadAt(buf, newEnd)
			check(err)
		}
		newStart += 1
		log.Printf("start: %d -> %d, end: %d -> %d", chunk.start, newStart, chunk.end, newEnd)
		chunk.start = newStart
		chunk.end = newEnd
	}
}

func MergeMap(left map[string]*MeasurementStats, right map[string]*MeasurementStats) {
	for k, v := range right {
		mergeStats(left[k], v)
	}
}

func HandleMmap(file string) string {
	start := time.Now()
	cmp := time.Now()
	reader, err := mmap.Open(file)
	check(err)
	defer reader.Close()
	chunks := LengthToChunks(reader.Len(), runtime.NumCPU())
	AdjustChunks(reader, chunks)
	var wg sync.WaitGroup
	chunkMaps := make([]map[string]*MeasurementStats, len(chunks))
	log.Printf("Prepped for chunking (%s)", time.Since(cmp))
	cmp = time.Now()
	for i, chunk := range chunks {
		wg.Add(1)
		go func(rng *Range) {
			wcmp := time.Now()
			log.Printf("w%d starting (%s)", i, time.Since(cmp))
			defer wg.Done()
			buf := make([]byte, rng.end-rng.start)
			m := make(map[string]*MeasurementStats)
			currOffset := rng.start
			read, err := reader.ReadAt(buf, currOffset)
			check(err)
			if read < int(rng.end-rng.start) {
				log.Fatalf("Expected to read %d, read %d", read, rng.end-rng.start)
			}
			log.Printf("w%d prepped (%s)", i, time.Since(wcmp))
			wcmp = time.Now()
			byteBuffer := bytes.NewBuffer(buf)
			scanner := bufio.NewScanner(byteBuffer)
			log.Printf("w%d buffer created (%s)", i, time.Since(wcmp))
			wcmp = time.Now()
			for scanner.Scan() {
				text := scanner.Text()
				if text == "" {
					continue
				}
				city, value, _ := strings.Cut(text, ";")
				valueFloat, err := strconv.ParseFloat(value, 64)
				check(err)
				stats, found := m[city]
				if found {
					addStat(stats, valueFloat)
				} else {
					m[city] = &MeasurementStats{min: valueFloat, max: valueFloat, sum: valueFloat, count: 1}
				}
			}
			log.Printf("w%d done (%s)", i, time.Since(wcmp))
			chunkMaps[i] = m
		}(chunk)
	}
	wg.Wait()
	log.Printf("Handled chunks (%s)", time.Since(cmp))
	cmp = time.Now()
	merged := chunkMaps[0]
	for _, m := range chunkMaps[1:] {
		MergeMap(merged, m)
	}
	results := make([]string, 0)
	log.Printf("Merged result maps (%s)", time.Since(cmp))
	cmp = time.Now()
	for k, v := range merged {
		results = append(results, fmt.Sprintf("%s=%.1f/%.1f/%.1f", k, v.min, (v.sum/float64(v.count)), v.max))
	}
	log.Printf("Crunched results (%s)", time.Since(cmp))
	cmp = time.Now()
	slices.Sort(results)
	log.Printf("Sorted results (%s)", time.Since(cmp))
	log.Printf("Total duration: %s", time.Since(start))
	return fmt.Sprintf("{%s}", strings.Join(results, ", "))
}
