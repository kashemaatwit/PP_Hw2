#define PNG_NO_SETJMP

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <png.h>
#include <mpi.h>
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

void write_to_image(int width, int* image, int* tmp) {
    int j;

    for (j = 0; j < width; j++)
        image[(int)(tmp[0] * width + j)] = tmp[j+1];
}

void barrier(int* count, int numThread) {
    #pragma omp critical (barrier)
    {
        *count = (*count)%(numThread) + 1;
    }

    while (*count < numThread) {
        #pragma omp flush
    }
}

int main(int argc, char* argv[]) {
	int i, j, id, numThread, width, height, intensity, iterationMax, myRank, size, init, activeNode, otherNode;
	int barrier1, barrier2, barrier3, sharedFlag, slaveFlag, masterFlag, slaveJ;
    double x, y, x0, y0, tmp, left, right, lower, upper, distance;
	char* output;
    MPI_Status status;
    MPI_Request request;

	// ===== init argument ===== //
	numThread = atoi(argv[1]);
	left = atof(argv[2]);
	right = atof(argv[3]);
	lower = atof(argv[4]);
	upper = atof(argv[5]);
	width = atoi(argv[6]);
	height = atoi(argv[7]);
	output = argv[8];
	iterationMax = 100000;

	// ===== allocate memory for image ===== //
	int* image = (int*)malloc(width * height * sizeof(int));
    int* bufferM = (int*)malloc((width * 1 + 1) * sizeof(int));
    int* bufferS = (int*)malloc((width * 1 + 1) * sizeof(int));
    assert(image);
    assert(bufferM);
    assert(bufferS);

    // Start MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ===== Master process ===== //
    if (myRank == 0) {

        // check whether master process need to do slave
        if (numThread > 1)
            init = 0;
        else
            init = 1;

        barrier1 = 0;
        barrier2 = 0;
        barrier3 = 0;
        otherNode = 0;
        activeNode = 0;
        sharedFlag = 0;
        #pragma omp parallel num_threads(numThread) private(i, id, slaveJ, x0, x, y, distance, intensity, tmp)
        {
            id = omp_get_thread_num();

            // ====== master thread do master job ====== //
            #pragma omp master
            {
                // ====== master thread assign job ====== //
                for (i = 0; i < height; i++) {
                    if (init < size) {
                        // init job to slave 0
                        if (init == 0) {
                            #pragma omp critical (rw_sharedFlag)
                                sharedFlag = i;
                        }
                        // init job to other slave
                        else {
                            MPI_Isend(&i, 1, MPI_INT, init, 0, MPI_COMM_WORLD, &request);
                            otherNode++;
                        }

                        init++;
                        activeNode++;
                    }
                    else {
                        masterFlag = 0;
                        #pragma omp critical (rw_sharedFlag)
                            masterFlag = sharedFlag;

                        // only one process
                        if (size == 1) {
                            while (masterFlag != -1) {
                                #pragma omp critical (rw_sharedFlag)
                                    masterFlag = sharedFlag;
                            }
                        }

                        // check whether slave 0 finish
                        if (masterFlag == -1) {
                            #pragma omp critical (rw_sharedFlag)
                            {
                                sharedFlag = i;
                                memcpy(bufferM, bufferS, (width+1) * sizeof(int));
                            }
                        }
                        // check whether other slave done
                        else if (otherNode > 0) {
                            MPI_Recv(bufferM, width+1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
                            MPI_Isend(&i, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &request);
                        }

                        // deal with return value
                        write_to_image(width, image, bufferM);
                    }

                }

                // ====== master thread wait for response ====== //
                while (activeNode != 0) {
                    i = -1;
                    masterFlag = 0;

                    #pragma omp critical (rw_sharedFlag)
                        masterFlag = sharedFlag;

                    if (masterFlag != -2 && masterFlag == -1) {
                        #pragma omp critical (rw_sharedFlag)
                            sharedFlag = -2;

                        activeNode--;

                        // deal with return value
                        write_to_image(width, image, bufferS);
                    }
                    else if (otherNode > 0) {
                        MPI_Recv(bufferM, width+1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
                        MPI_Isend(&i, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &request);

                        otherNode--;
                        activeNode--;

                        // deal with return value
                        write_to_image(width, image, bufferM);
                    }
                }
            }
            // ====== master thread end ====== //


            // ====== other thread do slave job ====== //
            if (id != 0) {

                // ====== slave thread wait for init job ====== //
                if (id == 1) {
                    slaveFlag = -1;

                    while (slaveFlag == -1) {
                        #pragma omp critical (rw_sharedFlag)
                            slaveFlag = sharedFlag;
                    }
                }

                // wait for init job
                barrier(&barrier1, numThread-1);

                while (slaveFlag != -2)
                {
                    // ====== slave thread init job config ====== //
                    if (id == 1) {
                        bufferS[0] = slaveFlag;

                        y0 = slaveFlag * ((upper-lower)/height) + lower;
                        j = 0;
                    }

                    // wait for init job config
                    barrier(&barrier2, numThread-1);

                    // each thread get the job ticket
                    #pragma omp critical (getTicket)
                    {
                        slaveJ = j;
                        j++;
                    }

                    // ====== slave thread do Mandelbrot test ====== //
                    while (slaveJ < width) {
                        x0 = slaveJ * ((right-left)/width) + left;

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
                        bufferS[slaveJ+1] = intensity;

                        // each thread get the job ticket again
                        #pragma omp critical (getTicket)
                        {
                            slaveJ = j;
                            j++;
                        }
                    }

                    // wait for everyone finish job
                    barrier(&barrier3, numThread-1);

                    // ====== slave thread return job and get next job ====== //
                    if (id == 1)
                    {
                        #pragma omp critical (rw_sharedFlag)
                            sharedFlag = -1;

                        slaveFlag = -1;
                        while (slaveFlag == -1) {
                            #pragma omp critical (rw_sharedFlag)
                                slaveFlag = sharedFlag;
                        }
                    }

                    // wait for another job
                    barrier(&barrier1, numThread-1);
                }
            }
            // ====== other thread end ====== //
        }

        write_png(output, width, height, image);
    }
    // ===== Master process end ===== //


    // ===== Slave process ===== //
    if (myRank != 0 && height >= myRank) {
        if (numThread > 1 && height <= myRank)
            printf("no action\n");
        else {

            MPI_Recv(&i, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

            while (i != -1) {
                bufferS[0] = i;
                y0 = i * ((upper-lower)/height) + lower;

                #pragma omp parallel num_threads(numThread) private(j, x0, x, y, distance, intensity, tmp)
                {
                    #pragma omp for schedule(dynamic)
                    for (j = 0; j < width; j++) {
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
                        bufferS[j+1] = intensity;
                    }
                }

                MPI_Send(bufferS, width+1, MPI_INT, 0, 1, MPI_COMM_WORLD);
                MPI_Recv(&i, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            }
        }
    }
    // ===== Slave process end ===== //
    MPI_Finalize();
    free(image);
    free(bufferS);
    free(bufferM);

	return 0;
}
