GCC=gcc
CFLAGSGCC=-Wall -march=native -mtune=native -std=c18 -I.

SIZE=128
PAD=2

all: nopad padk padj allpad

benchmark: nopad padk padj

nopad:
	$(GCC) nopad.c $(CFLAGSGCC) -O3  -lm -D SIZE=$(SIZE) -o $@	

padk:
	$(GCC) padk.c $(CFLAGSGCC) -O3  -lm -D SIZE=$(SIZE) -D PAD=$(PAD) -o $@	

padj:
	$(GCC) padj.c $(CFLAGSGCC) -O3  -lm -D SIZE=$(SIZE) -D PAD=$(PAD) -o $@	

allpad:
	$(GCC) allpad.c $(CFLAGSGCC) -O3  -lm -D SIZE=$(SIZE) -o $@	


run: benchmark
	./nopad
	./padk
	./padj

clean:
	rm -f *.o \
	nopad padk padj allpad \

.PHONY: clean all benchmark run
