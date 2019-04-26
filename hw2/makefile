CXX = g++
CXXFLAGS = -std=c++11 -O3 -march=native -fopenmp

CC = gcc
CFLAGS = -O3 -fopenmp

TARGETS = $(basename $(wildcard *.cpp)) $(basename $(wildcard *.c))

all : $(TARGETS)

%.c:
	$(CC) $(CFLAGS) $< $(LIBS) -o $@

%.cpp:
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@

clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
