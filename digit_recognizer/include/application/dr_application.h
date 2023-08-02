#ifndef DR_APPLICATION_H
#define DR_APPLICATION_H

#include <stdbool.h>

void dr_application_create();

void dr_application_close();

void dr_application_start();

bool dr_application_mnist_to_dataset(const char* images, const char* labels, const char* dataset);

#endif // DR_APPLICATION_H