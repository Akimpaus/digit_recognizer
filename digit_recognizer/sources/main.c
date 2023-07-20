#include <application/dr_application.h>
#include <stdio.h>

int main() {

    printf("Test\n");
    dr_application_create();
    dr_application_start();
    dr_application_close();

    return 0;
}