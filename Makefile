# choose compiler
CC=g++
# set flags
LIBS= -lOpenCL -lclblast
FLAGS= -Wall -Wunused-value -std=c++11

# start
all: cg

cg: test.cpp
	$(CC) $(FLAGS) $(LIBS) test.cpp -o cg.out


clean:
	rm -rf *.out
