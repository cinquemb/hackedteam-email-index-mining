CFLAGS= -Wall -Os -O3 -std=gnu++11 -pedantic `pkg-config --cflags --libs icu-uc icu-io` `pkg-config --cflags --libs libxml-2.0` `pkg-config --cflags --libs eigen3` /usr/local/lib/libarmadillo.dylib

all:
	ccache clang++ mine_emails.cpp -o mine_emails $(CFLAGS)

debug:
	ccache clang++ -v -g mine_emails.cpp -o mine_emails $(CFLAGS)

clean:
	rm mine_emails