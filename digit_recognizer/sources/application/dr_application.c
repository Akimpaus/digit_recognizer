#include <application/dr_application.h>
#include <application/dr_gui.h>

#define DR_APPLICATION_WINDOW_WIDTH         800
#define DR_APPLICATION_WINDOW_HEIGHT        600
#define DR_APPLICATION_DIGIT_RECOGNIZER_STR "Digit recognizer"
#define DR_APPLICATION_TAB_NAMES            "Dataset;Trainig;Prediction"
#define DR_APPLICATION_TAB_COUNT            3
#define DR_APPLICATION_TAB_HEIGHT           40
#define DR_APPLICATION_TAB_BOTTOM           DR_APPLICATION_TAB_HEIGHT
#define DR_APPLICATION_STATUS_BAR_HEIGHT          20
#define DR_APPLICATION_CANVAS_RESOLUTION_WIDTH    30
#define DR_APPLICATION_CANVAS_RESOLUTION_HEIGHT   30
#define DR_APPLICATION_CANVAS_PIXELS_COUNT        DR_APPLICATION_CANVAS_RESOLUTION_WIDTH *\
    DR_APPLICATION_CANVAS_RESOLUTION_HEIGHT
#define DR_APPLICATION_CANVAS_WIDTH               300
#define DR_APPLICATION_CANVAS_HEIGHT              300
#define DR_APPLICATION_CANVAS_DRAW_COLOR          WHITE
#define DR_APPLICATION_CANVAS_ERASE_COLOR         BLACK
#define DR_APPLICATION_CANVAS_CLEAR_BUTTON_HEIGHT 30
#define DR_APPLICATION_DIGITS_COUNT               10

typedef enum {
    dr_application_tab_dataset,
    dr_application_tab_training,
    dr_application_tab_prediction
} dr_application_tab;

Vector2 window_size                     = { 0, 0 };
Rectangle work_area                     = { 0 };
dr_application_tab current_tab          = dr_application_tab_dataset;
RenderTexture2D dataset_canvas_rtexture = { 0 };
Vector2 dataset_canvas_last_point       = { -1 };
char dataset_status_bar_str_buffer[DR_STR_BUFFER_SIZE]            = DR_APPLICATION_DIGIT_RECOGNIZER_STR;
size_t dataset_digits_count[DR_APPLICATION_DIGITS_COUNT]          = { 0 };
DR_FLOAT_TYPE* dataset_digits_pixels[DR_APPLICATION_DIGITS_COUNT] = { NULL };

void dr_application_dataset_add_digit(const size_t digit) {
    DR_ASSERT_MSG(digit >= 0 && digit <= 9, "attempt to add a not correct digit to application dataset");
    const size_t old_digits_count = dataset_digits_count[digit];
    const size_t new_digits_count = old_digits_count + 1;
    DR_FLOAT_TYPE* reallocated_pixels = (DR_FLOAT_TYPE*)DR_REALLOC(dataset_digits_pixels[digit],
        sizeof(DR_FLOAT_TYPE) * new_digits_count * DR_APPLICATION_CANVAS_PIXELS_COUNT);
    DR_ASSERT_MSG(reallocated_pixels, "new_pixels allocate error for application dataset");

    Image dataset_canvas_image = LoadImageFromTexture(dataset_canvas_rtexture.texture);
    DR_FLOAT_TYPE* new_pixels = reallocated_pixels + old_digits_count * DR_APPLICATION_CANVAS_PIXELS_COUNT;
    for (size_t y = DR_APPLICATION_CANVAS_RESOLUTION_HEIGHT; y > 0; --y) {
        for (size_t x = 0; x < DR_APPLICATION_CANVAS_RESOLUTION_WIDTH; ++x) {
            const Color pixel_color = GetImageColor(dataset_canvas_image, x, y - 1);
            *new_pixels = (float)(pixel_color.r + pixel_color.g + pixel_color.b) / (255.0f * 3.0f);
            ++new_pixels;
        }
    }

    UnloadImage(dataset_canvas_image);
    dataset_digits_count[digit]  = new_digits_count;
    dataset_digits_pixels[digit] = reallocated_pixels;
}

void dr_application_dataset_clear() {
    for (size_t i = 0; i < DR_APPLICATION_DIGITS_COUNT; ++i) {
        if (dataset_digits_count[i] > 0) {
            DR_FREE(dataset_digits_pixels[i]);
            dataset_digits_pixels[i] = NULL;
            dataset_digits_count[i] = 0;
        }
    }
}

void dr_application_dataset_canvas_clear() {
    BeginTextureMode(dataset_canvas_rtexture);
    ClearBackground(DR_APPLICATION_CANVAS_ERASE_COLOR);
    EndTextureMode();
}

void dr_application_create() {
    InitWindow(DR_APPLICATION_WINDOW_WIDTH, DR_APPLICATION_WINDOW_HEIGHT, DR_APPLICATION_DIGIT_RECOGNIZER_STR);
    SetTargetFPS(30);

    dataset_canvas_rtexture = LoadRenderTexture(
        DR_APPLICATION_CANVAS_RESOLUTION_WIDTH, DR_APPLICATION_CANVAS_RESOLUTION_HEIGHT);
    dr_application_dataset_canvas_clear();
}

void dr_application_close() {
    dr_application_dataset_clear();
    UnloadRenderTexture(dataset_canvas_rtexture);
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
    dataset_canvas_last_point = dr_gui_canvas(
        canvas_bounds, DR_APPLICATION_CANVAS_CLEAR_BUTTON_HEIGHT, dataset_canvas_rtexture,
        DR_APPLICATION_CANVAS_DRAW_COLOR, DR_APPLICATION_CANVAS_ERASE_COLOR, dataset_canvas_last_point);

    const Rectangle numeric_buttons_bounds = {
        work_area.x,
        canvas_bounds.y + canvas_bounds.height + DR_APPLICATION_CANVAS_CLEAR_BUTTON_HEIGHT,
        work_area.width,
        work_area.height - (canvas_bounds.y + canvas_bounds.height + DR_APPLICATION_CANVAS_CLEAR_BUTTON_HEIGHT)
    };
    const int clicked = dr_gui_numeric_buttons_row(
        numeric_buttons_bounds, DR_APPLICATION_DIGITS_COUNT, dataset_digits_count);
    if (clicked >= 0) {
        dr_application_dataset_add_digit(clicked);
        dr_application_dataset_canvas_clear();
        sprintf(dataset_status_bar_str_buffer, "%s %d %s", "Digit", clicked, "was added");
    }

    const Rectangle status_bar_rect = {
        0,
        window_size.y - DR_APPLICATION_STATUS_BAR_HEIGHT,
        window_size.x / 1.2,
        DR_APPLICATION_STATUS_BAR_HEIGHT
    };
    GuiStatusBar(status_bar_rect, dataset_status_bar_str_buffer);

    const float clear_dataset_button_rect_offset = 1;
    const Rectangle clear_dataset_button_rect = {
        status_bar_rect.x + status_bar_rect.width + clear_dataset_button_rect_offset,
        status_bar_rect.y,
        window_size.x - (status_bar_rect.x + status_bar_rect.width) - clear_dataset_button_rect_offset * 2,
        status_bar_rect.height
    };
    if (GuiButton(clear_dataset_button_rect, "Clear dataset")) {
        dr_application_dataset_clear();
        sprintf(dataset_status_bar_str_buffer, "%s", "Dataset has been cleared");
    }
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
        DR_ASSERT_MSG(false, "attempt to call the draw function for unknown application tab");
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