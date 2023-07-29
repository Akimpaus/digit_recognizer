#include <application/dr_application.h>

int main() {
    dr_application_create();
    dr_application_start();
    dr_application_close();
    return 0;
}


// #include <stdio.h>
// #include <stdint.h>
// #include <stdlib.h>
// #include <stdbool.h>

// #define PIXEL_COUNT (28 * 28)
// #define IMAGE_COUNT 60000

// typedef struct {
//     int32_t header[3];
//     float* data;
// } pixels;

// typedef struct {
//     int32_t header;
//     unsigned char* data;
// } labels;

// pixels* read_pixels(const char* file_path) {
//     FILE* file = fopen(file_path,  "rb");
//     if (!file) {
//         return NULL;
//     }

//     pixels* result = (pixels*)malloc(sizeof(pixels));
//     const size_t size = sizeof(float) * PIXEL_COUNT * IMAGE_COUNT;
//     result->data = (float*)malloc(size);
//     fread(result->header, sizeof(result->header), 1, file);
//     fread(result->data, size, 1, file);

//     fclose(file);
//     return result;
// }

// labels* read_labels(const char* file_path) {
//     FILE* file = fopen(file_path,  "rb");
//     if (!file) {
//         return NULL;
//     }

//     labels* result = (labels*)malloc(sizeof(labels));
//     const size_t size = sizeof(unsigned char) * IMAGE_COUNT;
//     result->data = (unsigned char*)malloc(size);
//     fread(&result->header, sizeof(result->header), 1, file);
//     fread(result->data, size, 1, file);

//     fclose(file);
//     return result;
// }

// bool write_dataset() {
//     pixels* pixels = read_pixels("/home/akim/dataset_pixels.bin");
//     labels* labels = read_labels("//home/akim/dataset_labels.bin");

//     if (!pixels || !labels) {
//         printf("error\n");
//         return false;
//     }

//     FILE* file = fopen("/home/akim/dataset.bin", "wb");
//     if (!file) {
//         printf("write error\n");
//         return false;
//     }

//     int32_t header[3] = { 60000, 28, 28 };
//     fwrite(header, sizeof(header), 1, file);
//     for (size_t i = 0; i < PIXEL_COUNT * IMAGE_COUNT; ++i) {
//         unsigned char val = (pixels->data[i] > 0.5 ? 1 : 0);
//         fwrite(&val, sizeof(unsigned char), 1, file);
//     }

//     fwrite(labels->data, sizeof(unsigned char) * IMAGE_COUNT, 1, file);

//     fclose(file);
//     return true;
// }

// void print_dataset(size_t count) {
//     FILE* file = fopen("/home/akim/dataset.bin", "rb");
//     if (!file) {
//         printf("error print dataset\n");
//         return;
//     }

//     int32_t header[3] = { 0 };
//     unsigned char* pixels = (unsigned char*)malloc(sizeof(unsigned char) * count * PIXEL_COUNT);
//     unsigned char* labels = (unsigned char*)malloc(sizeof(unsigned char) * count);

//     fread(header, sizeof(header), 1, file);
//     fread(pixels, sizeof(unsigned char) * count * PIXEL_COUNT, 1, file);

//     fseek(file, sizeof(header) + (sizeof(unsigned char) * IMAGE_COUNT * PIXEL_COUNT), SEEK_SET);
//     fread(labels, sizeof(unsigned char) * count, 1, file);

//     for (size_t i = 0; i < count * PIXEL_COUNT; ++i) {
//         if (i > 0 && i % 28 == 0) {
//             printf("\n");
//         }
//         printf("%s", (pixels[i] == 1 ? "#" : " "));
//     }

//     for (size_t i = 0; i < count; ++i) {
//         printf("%d\n", labels[i]);
//     }

//     fclose(file);
// }

// int main() {

//     write_dataset();
//     print_dataset(10);

//     return 1;
// }