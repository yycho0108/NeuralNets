
COMPILER = g++
FLAGS = -std=c++11
DEBUGFLAGS = -g -Wall
LIBS = -L/usr/lib/ -larmadillo

DEBUGFLAGS = -g
all : main.cpp
	$(COMPILER) $(FLAGS) -O3 main.cpp $(LIBS) -o main.o
debug : main.cpp
	$(COMPILER) $(FLAGS) $(DEBUGFLAGS) main.cpp $(LIBS) -o debug.o
clean :
	rm -rf *.o

