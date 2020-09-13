GCC=gcc
CFLAGSGCC=-Wall -march=native -mtune=native -std=c18 -I.

THREADS=4
SIZE=256
TSIZE=4
KSPLIT=4

all: benchmark

benchmark:
	$(GCC) snippet.c $(CFLAGSGCC) -O3  -lm -fopenmp -D THREADS=$(THREADS) -D SIZE=$(SIZE) -D TSIZE=$(TSIZE) -D KSPLIT=$(KSPLIT) -o $@	

run: benchmark
	./benchmark

clean:
	rm -f *.o \
	benchmark \

.PHONY: clean all run
