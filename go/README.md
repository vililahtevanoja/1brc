# 1BRC with Go

### Running

Install dependencies:

```go get .```

Run tests:
```go test```

Run application:
```go run .```

Run application with profiling:
```go run . pprof```

View profile (requires graphviz, run e.g. `brew install graphviz` beforehand):

```go tool pprof -http=":8000" 1brc-go ./profile/cpu.pprof```
