CFLAGS= -Wall -Os -std=gnu++11 -pedantic `pkg-config --cflags --libs eigen3`

all:
	ccache clang++ generate_knn_plots.cpp -o generate_knn_plots $(CFLAGS)

debug:
	ccache clang++ -v -g generate_knn_plots.cpp -o generate_knn_plots $(CFLAGS)

clean:
	rm generate_knn_plots