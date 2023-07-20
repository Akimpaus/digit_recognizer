#include <application/dr_application.h>
#include <application/dr_gui.h>

#define DR_APPLICATION_DIGIT_RECOGNIZER_STR "Digit recognizer"
#define DR_APPLICATION_TAB_NAMES     "Dataset;Trainig;Prediction"
#define DR_APPLICATION_TAB_COUNT     3
#define DR_APPLICATION_TAB_HEIGHT    40
#define DR_APPLICATION_TAB_BOTTOM    DR_APPLICATION_TAB_HEIGHT
#define DR_APPLICATION_STATUS_BAR_HEIGHT 20
#define DR_APPLICATION_CANVAS_RESOLUTION_WIDTH  30
#define DR_APPLICATION_CANVAS_RESOLUTION_HEIGHT 30
#define DR_APPLICATION_CANVAS_WIDTH  300
#define DR_APPLICATION_CANVAS_HEIGHT 300
#define DR_APPLICATION_CANVAS_DRAW_COLOR  WHITE
#define DR_APPLICATION_CANVAS_ERASE_COLOR BLACK
#define DR_APPLICATION_CANVAS_CLEAR_BUTTON_HEIGHT 30

typedef enum {
    dr_application_tab_dataset,
    dr_application_tab_training,
    dr_application_tab_prediction
} dr_application_tab;

Vector2 window_size               = { 0, 0 };
Rectangle work_area               = { 0 };
dr_application_tab current_tab    = dr_application_tab_dataset;
RenderTexture2D canvas_dataset    = { 0 };
Vector2 canvas_dataset_last_point = { -1 };
char dataset_status_bar_str_buffer[DR_STR_BUFFER_SIZE] = DR_APPLICATION_DIGIT_RECOGNIZER_STR;

void dr_application_create() {
    InitWindow(800, 600, DR_APPLICATION_DIGIT_RECOGNIZER_STR);
    SetTargetFPS(30);

    canvas_dataset = LoadRenderTexture(DR_APPLICATION_CANVAS_RESOLUTION_WIDTH, DR_APPLICATION_CANVAS_RESOLUTION_HEIGHT);


    BeginTextureMode(canvas_dataset);
    ClearBackground(DR_APPLICATION_CANVAS_ERASE_COLOR);
    EndTextureMode();
}

void dr_application_close() {
    UnloadRenderTexture(canvas_dataset);
    CloseWindow();
}

void dr_application_update_dataset_tab() {
    work_area = CLITERAL(Rectangle) {
        0,
        DR_APPLICATION_TAB_BOTTOM,
        window_size.x,
        window_size.y - DR_APPLICATION_STATUS_BAR_HEIGHT
    };
}

void dr_application_update_training_tab() {
}

void dr_application_update_prediction_tab() {
}

void dr_application_draw_dataset_tab() {
    const Rectangle canvas_bounds = {
        work_area.x + (work_area.width / 2 - DR_APPLICATION_CANVAS_WIDTH / 2),
        work_area.y + (work_area.height / 2 - DR_APPLICATION_CANVAS_HEIGHT / 1.3),
        DR_APPLICATION_CANVAS_WIDTH,
        DR_APPLICATION_CANVAS_HEIGHT - DR_APPLICATION_CANVAS_CLEAR_BUTTON_HEIGHT / 2
    };
    canvas_dataset_last_point = dr_gui_canvas(canvas_bounds, DR_APPLICATION_CANVAS_CLEAR_BUTTON_HEIGHT, canvas_dataset,
        DR_APPLICATION_CANVAS_DRAW_COLOR, DR_APPLICATION_CANVAS_ERASE_COLOR, canvas_dataset_last_point);

    const Rectangle numeric_buttons_bounds = {
        work_area.x,
        canvas_bounds.y + canvas_bounds.height + DR_APPLICATION_CANVAS_CLEAR_BUTTON_HEIGHT,
        work_area.width,
        work_area.height - (canvas_bounds.y + canvas_bounds.height + DR_APPLICATION_CANVAS_CLEAR_BUTTON_HEIGHT)
    };
    const int clicked = dr_gui_numeric_buttons_row(numeric_buttons_bounds, 10, NULL);
    if (clicked >= 0) {
        BeginTextureMode(canvas_dataset);
        ClearBackground(DR_APPLICATION_CANVAS_ERASE_COLOR);
        EndTextureMode();
        sprintf(dataset_status_bar_str_buffer, "%s %d %s", "Digit", clicked, "was added");
    }

    Rectangle status_bar_rect = {
        0,
        window_size.y - DR_APPLICATION_STATUS_BAR_HEIGHT,
        window_size.x,
        DR_APPLICATION_STATUS_BAR_HEIGHT
    };
    GuiStatusBar(status_bar_rect, dataset_status_bar_str_buffer);
}

void dr_application_draw_training_tab() {
}

void dr_application_draw_prediction_tab() {
}

void dr_application_update() {
    window_size = CLITERAL(Vector2){ (float)GetScreenWidth(), (float)GetScreenHeight() };

    switch (current_tab) {
    case dr_application_tab_dataset:
        dr_application_update_dataset_tab();
        break;
    case dr_application_tab_training:
        dr_application_update_training_tab();
        break;
    case dr_application_tab_prediction:
        dr_application_update_prediction_tab();
        break;
    default:
        DR_ASSERT_MSG(false, "attempt to call the update function for unknown application tab");
        break;
    }
}

void dr_application_draw() {
    const Rectangle tab_rect = { 0, 0, window_size.x / DR_APPLICATION_TAB_COUNT, DR_APPLICATION_TAB_HEIGHT };
    current_tab = GuiToggleGroup(tab_rect, DR_APPLICATION_TAB_NAMES, current_tab);

    switch (current_tab) {
    case dr_application_tab_dataset:
        dr_application_draw_dataset_tab();
        break;
    case dr_application_tab_training:
        dr_application_draw_training_tab();
        break;
    case dr_application_tab_prediction:
        dr_application_draw_prediction_tab();
        break;
    default:
        DR_ASSERT_MSG(false, "attempt to call the update function for unknown application tab");
        break;
    }
}

void dr_application_start() {
    while (!WindowShouldClose()) {
        dr_application_update();

        BeginDrawing();
        ClearBackground(CLITERAL(Color){ 20, 20, 25, 255 });
        dr_application_draw();
        EndDrawing();
    }
}