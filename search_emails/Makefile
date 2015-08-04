CFLAGS= -Wall -Os -O3 -std=gnu++11 -pedantic `pkg-config --cflags --libs eigen3` /usr/local/lib/libarmadillo.dylib

all:
	ccache clang++ search_emails.cpp -o search_emails $(CFLAGS)

debug:
	ccache clang++ -v -g search_emails.cpp -o search_emails $(CFLAGS)

clean:
	rm search_emails