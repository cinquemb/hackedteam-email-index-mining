CFLAGS= -Wall -Os -O3 -std=gnu++11 -pedantic `pkg-config --cflags --libs eigen3` /usr/local/lib/libarmadillo.dylib

all:
	ccache clang++ create_right_hand_uv.cpp -o create_right_hand_uv $(CFLAGS)

debug:
	ccache clang++ -v -g create_right_hand_uv.cpp -o create_right_hand_uv $(CFLAGS)

clean:
	rm create_right_hand_uv