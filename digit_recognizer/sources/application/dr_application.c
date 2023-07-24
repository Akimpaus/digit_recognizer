#include <application/dr_application.h>
#include <application/dr_gui.h>

#define DR_APPLICATION_WINDOW_WIDTH         800
#define DR_APPLICATION_WINDOW_HEIGHT        600
#define DR_APPLICATION_DIGIT_RECOGNIZER_STR "Digit recognizer"
#define DR_APPLICATION_TAB_COUNT            3
#define DR_APPLICATION_TAB_HEIGHT           40
#define DR_APPLICATION_TAB_BOTTOM           DR_APPLICATION_TAB_HEIGHT
#define DR_APPLICATION_STATUS_BAR_HEIGHT    20
#define DR_APPLICATION_DIGITS_COUNT         10

#define DR_APPLICATION_CANVAS_RESOLUTION_WIDTH    30
#define DR_APPLICATION_CANVAS_RESOLUTION_HEIGHT   30
#define DR_APPLICATION_CANVAS_PIXELS_COUNT        DR_APPLICATION_CANVAS_RESOLUTION_WIDTH *\
    DR_APPLICATION_CANVAS_RESOLUTION_HEIGHT
#define DR_APPLICATION_CANVAS_WIDTH               300
#define DR_APPLICATION_CANVAS_HEIGHT              300
#define DR_APPLICATION_CANVAS_DRAW_COLOR          WHITE
#define DR_APPLICATION_CANVAS_ERASE_COLOR         BLACK
#define DR_APPLICATION_CANVAS_CLEAR_BUTTON_HEIGHT 30
#define DR_APPLICATION_CANVAS_WINDOW_HEIGHT       DR_APPLICATION_CANVAS_HEIGHT +\
    DR_APPLICATION_CANVAS_CLEAR_BUTTON_HEIGHT

typedef enum {
    dr_application_tab_dataset,
    dr_application_tab_training,
    dr_application_tab_prediction
} dr_application_tab;

// general
Vector2 window_size            = { 0 };
Rectangle work_area            = { 0 };
dr_application_tab current_tab = dr_application_tab_dataset;

// dataset
RenderTexture2D dataset_canvas_rtexture = { 0 };
Vector2 dataset_canvas_last_point       = { -1 };
char dataset_status_bar_str_buffer[DR_STR_BUFFER_SIZE]            = DR_APPLICATION_DIGIT_RECOGNIZER_STR;
size_t dataset_digits_count[DR_APPLICATION_DIGITS_COUNT]          = { 0 };
DR_FLOAT_TYPE* dataset_digits_pixels[DR_APPLICATION_DIGITS_COUNT] = { NULL };

// training
int training_list_view_index          = 0;
int training_list_view_active         = 0;
int training_controller_layer_size    = 1;
int training_controller_dropbox_index = 0;
bool training_controller_dropbox_edit = false;


void dr_application_dataset_add_digit(const size_t digit) {
    DR_ASSERT_MSG(digit >= 0 && digit <= 9, "attempt to add a not correct digit to application dataset");
    const size_t old_digits_count     = dataset_digits_count[digit];
    const size_t new_digits_count     = old_digits_count + 1;
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
            dataset_digits_count[i]  = 0;
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
        window_size.y - DR_APPLICATION_STATUS_BAR_HEIGHT - DR_APPLICATION_TAB_BOTTOM
    };
}

void dr_application_update_training_tab() {
    work_area = CLITERAL(Rectangle) {
        0,
        DR_APPLICATION_TAB_BOTTOM,
        window_size.x,
        window_size.y - DR_APPLICATION_TAB_BOTTOM
    };
}

void dr_application_update_prediction_tab() {
}

void dr_application_draw_dataset_tab() {
    const Rectangle canvas_bounds = {
        work_area.x + (work_area.width / 2 - DR_APPLICATION_CANVAS_WIDTH / 2),
        work_area.y + (work_area.height / 2.2 - DR_APPLICATION_CANVAS_HEIGHT / 2) -
            DR_APPLICATION_CANVAS_CLEAR_BUTTON_HEIGHT,
        DR_APPLICATION_CANVAS_WIDTH,
        DR_APPLICATION_CANVAS_HEIGHT
    };
    dataset_canvas_last_point = dr_gui_canvas(
        canvas_bounds, DR_APPLICATION_CANVAS_CLEAR_BUTTON_HEIGHT, dataset_canvas_rtexture,
        DR_APPLICATION_CANVAS_DRAW_COLOR, DR_APPLICATION_CANVAS_ERASE_COLOR, dataset_canvas_last_point);

    const Rectangle numeric_buttons_bounds = {
        work_area.x,
        canvas_bounds.y + DR_APPLICATION_CANVAS_WINDOW_HEIGHT,
        work_area.width,
        work_area.height - (canvas_bounds.y + canvas_bounds.height)
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
    const bool layer_selected = training_list_view_active >= 0;

    const Vector2 hidden_layers_bounds_size = {
        work_area.width / 2,
        work_area.height / 1.6
    };
    const Rectangle hidden_layers_bounds = {
        work_area.x + work_area.width / 2 - hidden_layers_bounds_size.x / 2,
        work_area.y + work_area.height / 2.3 - hidden_layers_bounds_size.y / 2,
        hidden_layers_bounds_size.x,
        hidden_layers_bounds_size.y
    };

    const Vector2 list_view_bounds_size = {
        hidden_layers_bounds.width / 2,
        hidden_layers_bounds.height / 1.3
    };
    const Rectangle list_view_bounds = {
        hidden_layers_bounds.x + (hidden_layers_bounds.width / 2 - list_view_bounds_size.x / 2) * !layer_selected,
        hidden_layers_bounds.y,
        list_view_bounds_size.x,
        list_view_bounds_size.y
    };

    const Rectangle list_view_controller_bounds = {
        list_view_bounds.x,
        list_view_bounds.y + list_view_bounds.height,
        list_view_bounds.width,
        (hidden_layers_bounds.y + hidden_layers_bounds.height) - (list_view_bounds.y + list_view_bounds.height)
    };

    const float layer_controller_bounds_margin_h = 5;
    const Vector2 layer_controller_bounds_size = {
        hidden_layers_bounds.width - list_view_bounds.width - layer_controller_bounds_margin_h,
        hidden_layers_bounds.height / 6
    };
    const Rectangle layer_controller_bounds = {
        hidden_layers_bounds.x + hidden_layers_bounds.width - layer_controller_bounds_size.x +
            layer_controller_bounds_margin_h,
        hidden_layers_bounds.y + hidden_layers_bounds.height / 2 - layer_controller_bounds_size.y / 2,
        layer_controller_bounds_size.x,
        layer_controller_bounds_size.y
    };

    training_list_view_active = GuiListView(list_view_bounds, "test;test1;test2",
        &training_list_view_index, training_list_view_active);

    const Rectangle list_view_controller_toggle_group_bounds = {
        list_view_controller_bounds.x,
        list_view_controller_bounds.y,
        list_view_controller_bounds.width,
        list_view_controller_bounds.height / 3
    };
    GuiToggleGroup(list_view_controller_toggle_group_bounds, "Add\nRemove\nClear", -1);

    const Vector2 layer_controller_element_size = {
        layer_controller_bounds.width - layer_controller_bounds_margin_h,
        layer_controller_bounds.height / 2
    };
    if (layer_selected) {
        const Rectangle layer_controller_spinner_bounds = {
            layer_controller_bounds.x,
            layer_controller_bounds.y,
            layer_controller_element_size.x,
            layer_controller_element_size.y
        };
        GuiSpinner(layer_controller_spinner_bounds, NULL, &training_controller_layer_size, 1, 100, false);

        const Rectangle layer_controller_dropbox_bounds = {
            layer_controller_spinner_bounds.x,
            layer_controller_spinner_bounds.y + layer_controller_spinner_bounds.height,
            layer_controller_element_size.x,
            layer_controller_element_size.y
        };
        if (GuiDropdownBox(layer_controller_dropbox_bounds, "sigmoid;tanh;ReLU",
            &training_controller_dropbox_index, training_controller_dropbox_edit)) {
            training_controller_dropbox_edit = !training_controller_dropbox_edit;
        }
    }

    const float dist_to_bottom =
        (work_area.y + work_area.height) - (hidden_layers_bounds.y + hidden_layers_bounds.height);
    const Rectangle train_button_bounds = {
        hidden_layers_bounds.x + hidden_layers_bounds.width / 2 - layer_controller_element_size.x / 2,
        (hidden_layers_bounds.y + hidden_layers_bounds.height) +
            dist_to_bottom / 2 - layer_controller_element_size.y / 2,
        layer_controller_element_size.x,
        layer_controller_element_size.y
    };
    if (GuiButton(train_button_bounds, "Train")) {
        printf("train\n");
    }
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

    const Rectangle tab_rect = { 0, 0, window_size.x / DR_APPLICATION_TAB_COUNT, DR_APPLICATION_TAB_HEIGHT };
    current_tab = GuiToggleGroup(tab_rect, "Dataset;Trainig;Prediction", current_tab);
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