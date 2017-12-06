#define PNG_NO_SETJMP

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <png.h>
#include <omp.h>

void write_png(const char* filename, const int width, const int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            row[x * 3] = ((p & 0xf) << 4);
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char* argv[]) {
	int i, j, numThread, width, height, intensity, iterationMax, myRank, size;
	double x, y, x0, y0, tmp, left, right, lower, upper, distance;
	char* output;

	// init argument
	numThread = atoi(argv[1]);
	left = atof(argv[2]);
	right = atof(argv[3]);
	lower = atof(argv[4]);
	upper = atof(argv[5]);
	width = atoi(argv[6]);
	height = atoi(argv[7]);
	output = argv[8];
	iterationMax = 100000;

	// allocate memory for image
	int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    // Start mandelbrot set - from lower left
    #pragma omp parallel num_threads(numThread) private(i, j, x0, y0, x, y, distance, intensity, tmp)
    {
        #pragma omp for collapse(2) schedule(dynamic)
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                y0 = i * ((upper-lower)/height) + lower;
                x0 = j * ((right-left)/width) + left;

                x = 0;
                y = 0;
                distance = 0;
                intensity = 0;
                while (intensity < iterationMax && distance < 4.0) {
                    tmp = x * x - y * y + x0;
                    y = 2 * x * y + y0;
                    x = tmp;
                    distance = x * x + y * y;

                    intensity++;
                }
                image[i * width + j] = intensity;
            }
        }
    }

    // Draw & Clean up
    write_png(output, width, height, image);
    free(image);

	return 0;
}
