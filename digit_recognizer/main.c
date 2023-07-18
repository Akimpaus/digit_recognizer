#include <dr_application.h>
#include <raylib.h>

#define RAYGUI_IMPLEMENTATION
#include <raygui.h>

int main() {

    InitWindow(800, 600, "Test");


    SetTargetFPS(30);
    while (!WindowShouldClose()) {
        BeginDrawing();

        Rectangle rect = { 0, 0, 100, 100 };
        GuiButton(rect, "test");

        EndDrawing();
    }

    CloseWindow();

    return 0;
}