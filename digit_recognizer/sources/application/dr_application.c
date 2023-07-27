#include <application/dr_application.h>
#include <application/dr_gui.h>
#include <neural_network/dr_neural_network.h>
#include <limits.h>

// POSIX
#include <pthread.h>

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

#define DR_APPLICATION_TRAINING_SIGMOID_STR "sigmoid"
#define DR_APPLICATION_TRAINING_TANH_STR    "tanh"
#define DR_APPLICATION_TRAINING_RELU_STR    "ReLU"

typedef enum {
    dr_application_tab_dataset,
    dr_application_tab_training,
    dr_application_tab_prediction
} dr_application_tab;

// general
Vector2 window_size              = { 0 };
dr_application_tab current_tab   = dr_application_tab_dataset;
dr_neural_network neural_network = { 0 };

// dataset
RenderTexture2D dataset_canvas_rtexture = { 0 };
Vector2 dataset_canvas_last_point       = { -1 };
char dataset_status_bar_str_buffer[DR_STR_BUFFER_SIZE]   = DR_APPLICATION_DIGIT_RECOGNIZER_STR;
size_t dataset_digits_count_all                          = 0;
size_t dataset_digits_count[DR_APPLICATION_DIGITS_COUNT] = { 0 };
DR_FLOAT_TYPE* dataset_digits_pixels                     = NULL;
size_t* dataset_digits_results                           = NULL;

// training
int training_list_view_scroll_index = 0;
int training_list_view_active       = -1;
int training_list_view_focus        = 0;
size_t training_controller_layer_size    = 1;
bool training_controller_layer_size_edit = false;
int training_controller_dropbox_index = 0;
bool training_controller_dropbox_edit = false;
size_t training_hidden_layers_count   = 0;
char** training_hidden_layers_info    = NULL;
DR_FLOAT_TYPE training_learning_rate  = 0.01;
size_t training_epochs              = 100000;
size_t training_current_epoch       = 0;
bool training_epochs_value_box_edit = false;
bool training_process_active        = false;
bool training_procces_finished      = false;
bool training_proccess_cancelled    = false;
DR_FLOAT_TYPE training_error        = 0;
size_t training_current_dataset_index = 0;
pthread_t training_thread_id = { 0 };

//////////////////////////////////////////////////////////////////////////////////////////////////////////////// DATASET

void dr_application_dataset_add_digit(const size_t digit) {
    DR_ASSERT_MSG(digit >= 0 && digit <= 9, "attempt to add a not correct digit to the application dataset");

    const size_t new_digits_count = dataset_digits_count_all + 1;
    DR_FLOAT_TYPE* reallocated_pixels = (DR_FLOAT_TYPE*)DR_REALLOC(
        dataset_digits_pixels, sizeof(DR_FLOAT_TYPE) * new_digits_count * DR_APPLICATION_CANVAS_PIXELS_COUNT);
    DR_ASSERT_MSG(reallocated_pixels, "new pixels reallocate error for the application dataset");
    size_t* reallocated_results = (size_t*)DR_REALLOC(dataset_digits_results, sizeof(size_t) * new_digits_count);
    DR_ASSERT_MSG(reallocated_results, "results reallocate error for the application dataset");

    Image dataset_canvas_image = LoadImageFromTexture(dataset_canvas_rtexture.texture);
    DR_FLOAT_TYPE* new_pixels = reallocated_pixels + dataset_digits_count_all * DR_APPLICATION_CANVAS_PIXELS_COUNT;
    for (size_t y = DR_APPLICATION_CANVAS_RESOLUTION_HEIGHT; y > 0; --y) {
        for (size_t x = 0; x < DR_APPLICATION_CANVAS_RESOLUTION_WIDTH; ++x) {
            const Color pixel_color = GetImageColor(dataset_canvas_image, x, y - 1);
            *new_pixels = (float)(pixel_color.r + pixel_color.g + pixel_color.b) / (255.0f * 3.0f);
            ++new_pixels;
        }
    }

    reallocated_results[new_digits_count - 1] = digit;
    dataset_digits_results   = reallocated_results;
    dataset_digits_count_all = new_digits_count;
    dataset_digits_pixels    = reallocated_pixels;
    ++dataset_digits_count[digit];
    UnloadImage(dataset_canvas_image);
}

void dr_application_dataset_clear() {
    if (dataset_digits_count_all == 0) {
        return;
    }

    for (size_t i = 0; i < DR_APPLICATION_DIGITS_COUNT; ++i) {
        dataset_digits_count[i] = 0;
    }
    DR_FREE(dataset_digits_pixels);
    dataset_digits_pixels = NULL;
    DR_FREE(dataset_digits_results);
    dataset_digits_results = NULL;
    dataset_digits_count_all = 0;
}

void dr_application_dataset_canvas_clear() {
    BeginTextureMode(dataset_canvas_rtexture);
    ClearBackground(DR_APPLICATION_CANVAS_ERASE_COLOR);
    EndTextureMode();
}

void dr_application_dataset_tab() {
    const bool dataset_empty = dataset_digits_count_all == 0;

    // work area
    Rectangle work_area = {
        0,
        DR_APPLICATION_TAB_BOTTOM,
        window_size.x,
        window_size.y - DR_APPLICATION_STATUS_BAR_HEIGHT - DR_APPLICATION_TAB_BOTTOM
    };

    // canvas
    const Rectangle canvas_bounds = {
        work_area.x + (work_area.width / 2 - DR_APPLICATION_CANVAS_WIDTH / 2),
        work_area.y + (work_area.height / 2.2 - DR_APPLICATION_CANVAS_HEIGHT / 2) -
            DR_APPLICATION_CANVAS_CLEAR_BUTTON_HEIGHT,
        DR_APPLICATION_CANVAS_WIDTH,
        DR_APPLICATION_CANVAS_HEIGHT
    };

    // numeric buttons
    const Rectangle numeric_buttons_bounds = {
        work_area.x,
        canvas_bounds.y + DR_APPLICATION_CANVAS_WINDOW_HEIGHT,
        work_area.width,
        work_area.height - (canvas_bounds.y + canvas_bounds.height)
    };

    // status bar
    const Rectangle status_bar_rect = {
        0,
        window_size.y - DR_APPLICATION_STATUS_BAR_HEIGHT,
        window_size.x / (dataset_empty ? 1 : 1.2),
        DR_APPLICATION_STATUS_BAR_HEIGHT
    };

    // clear dataset button
    const float clear_dataset_button_rect_offset = 1;
    const Rectangle clear_dataset_button_rect = {
        status_bar_rect.x + status_bar_rect.width + clear_dataset_button_rect_offset,
        status_bar_rect.y,
        window_size.x - (status_bar_rect.x + status_bar_rect.width) - clear_dataset_button_rect_offset * 2,
        status_bar_rect.height
    };

    // gui
    dataset_canvas_last_point = dr_gui_canvas(
        canvas_bounds, DR_APPLICATION_CANVAS_CLEAR_BUTTON_HEIGHT, dataset_canvas_rtexture,
        DR_APPLICATION_CANVAS_DRAW_COLOR, DR_APPLICATION_CANVAS_ERASE_COLOR, dataset_canvas_last_point);

    const int clicked = dr_gui_numeric_buttons_row(
        numeric_buttons_bounds, DR_APPLICATION_DIGITS_COUNT, dataset_digits_count);
    if (clicked >= 0) {
        dr_application_dataset_add_digit(clicked);
        dr_application_dataset_canvas_clear();
        sprintf(dataset_status_bar_str_buffer, "%s %d %s", "Digit", clicked, "was added");
    }

    GuiStatusBar(status_bar_rect, dataset_status_bar_str_buffer);

    if (!dataset_empty && GuiButton(clear_dataset_button_rect, "Clear dataset")) {
        dr_application_dataset_clear();
        sprintf(dataset_status_bar_str_buffer, "%s", "Dataset has been cleared");
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////// TRAINING

void dr_application_training_add_hidden_layer(const size_t layer_size, const char* activation_function) {
    training_hidden_layers_info =
        (char**)DR_REALLOC(training_hidden_layers_info, sizeof(char*) * (training_hidden_layers_count + 1));
    DR_ASSERT_MSG(training_hidden_layers_info, "application hidden layers info realloc error");

    training_hidden_layers_info[training_hidden_layers_count] = (char*)DR_MALLOC(sizeof(char) * DR_STR_BUFFER_SIZE);
    DR_ASSERT_MSG(training_hidden_layers_info[training_hidden_layers_count],
        "application hidden layers info alloc error");

    sprintf(training_hidden_layers_info[training_hidden_layers_count], "%zu %s", layer_size, activation_function);
    ++training_hidden_layers_count;
}

void dr_application_training_remove_hidden_layer(const size_t index) {
    DR_ASSERT_MSG(index < training_hidden_layers_count, "attempt to remove a not exist hidden layer");

    const size_t new_layers_count = training_hidden_layers_count - 1;
    char** new_arr = (char**)DR_MALLOC(sizeof(char*) * new_layers_count);
    DR_ASSERT_MSG(new_arr, "apllication hidden layer remove - new hidden layers info alloc error");
    size_t new_arr_i = 0;
    for (size_t i = 0; i < training_hidden_layers_count; ++i) {
        if (i == index) {
            DR_FREE(training_hidden_layers_info[i]);
            continue;
        }
        new_arr[new_arr_i] = (char*)DR_MALLOC(sizeof(char) * DR_STR_BUFFER_SIZE);
        DR_ASSERT_MSG(new_arr[new_arr_i], "apllication hidden layer remove - new hidden layer info alloc error");
        memcpy(new_arr[new_arr_i], training_hidden_layers_info[i], DR_STR_BUFFER_SIZE);
        DR_FREE(training_hidden_layers_info[i]);
        ++new_arr_i;
    }
    DR_FREE(training_hidden_layers_info);

    training_hidden_layers_info  = new_arr;
    training_hidden_layers_count = new_layers_count;
}

void dr_application_training_set_hidden_layer(
    const size_t layer_index, const size_t layer_size, const char* activation_function) {
    DR_ASSERT_MSG(layer_index < training_hidden_layers_count,
        "index out of range when trying to set a hidden layer in application");

    const char* text = TextFormat("%zu %s", layer_size, activation_function);
    TextCopy(training_hidden_layers_info[layer_index], text);
}

void dr_application_training_get_hidden_layer(const size_t layer_index, size_t* layer_size, char* activation_function) {
    DR_ASSERT_MSG(layer_index < training_hidden_layers_count,
        "index out of range when trying to get a hidden layer in application");

    sscanf(training_hidden_layers_info[layer_index], "%zu %s", layer_size, activation_function);
}

void dr_application_training_hidden_layers_info_clear() {
    if (training_hidden_layers_count == 0) {
        return;
    }

    for (size_t i = 0; i < training_hidden_layers_count; ++i) {
        DR_FREE(training_hidden_layers_info[i]);
        training_hidden_layers_info[i] = NULL;
    }
    DR_FREE(training_hidden_layers_info);
    training_hidden_layers_info = NULL;
    training_hidden_layers_count = 0;
}

void dr_application_neural_network_create() {
    const size_t layers_count = training_hidden_layers_count + 2;
    const size_t activation_functions_count = layers_count - 1;

    size_t* layers_sizes = (size_t*)DR_MALLOC(sizeof(size_t) * layers_count);
    DR_ASSERT_MSG(layers_sizes, "application neural network layers sizes alloc error");
    dr_activation_function* activation_functions =
        (dr_activation_function*)DR_MALLOC(sizeof(dr_activation_function) * activation_functions_count);
    DR_ASSERT_MSG(activation_functions, "application neural network activation functions alloc error");
    dr_activation_function* activation_function_derivatives =
        (dr_activation_function*)DR_MALLOC(sizeof(dr_activation_function) * activation_functions_count);
    DR_ASSERT_MSG(activation_function_derivatives,
        "application neural network activation functions derivatives alloc error");

    layers_sizes[0] = DR_APPLICATION_CANVAS_PIXELS_COUNT;
    layers_sizes[layers_count - 1] = DR_APPLICATION_DIGITS_COUNT;
    activation_functions[activation_functions_count - 1] = dr_sigmoid;
    activation_function_derivatives[activation_functions_count - 1] = dr_sigmoid_derivative;

    for (size_t i = 1; i < layers_count - 1; ++i) {
        size_t curr_layer_size = 0;
        char curr_activation_function_str[DR_STR_BUFFER_SIZE] = { 0 };
        sscanf(training_hidden_layers_info[i - 1], "%zu %s", &curr_layer_size, curr_activation_function_str);

        layers_sizes[i] = curr_layer_size;

        const size_t prev_i = i - 1;
        dr_activation_function* curr_activation_function = activation_functions + prev_i;
        dr_activation_function* curr_activation_function_derivatives = activation_function_derivatives + prev_i;
        if (strcmp(curr_activation_function_str, DR_APPLICATION_TRAINING_SIGMOID_STR) == 0) {
            *curr_activation_function = dr_sigmoid;
            *curr_activation_function_derivatives = dr_sigmoid_derivative;
        } else if (strcmp(curr_activation_function_str, DR_APPLICATION_TRAINING_TANH_STR) == 0) {
            *curr_activation_function = dr_tanh;
            *curr_activation_function_derivatives = dr_tanh_derivative;
        }  else if (strcmp(curr_activation_function_str, DR_APPLICATION_TRAINING_RELU_STR) == 0) {
            *curr_activation_function = dr_relu;
            *curr_activation_function_derivatives = dr_relu_derivative;
        } else {
            DR_ASSERT_MSG(false, "unknown activation function in application");
        }
    }

    neural_network = dr_neural_network_create(
        layers_sizes, layers_count, activation_functions, activation_function_derivatives);

    DR_FREE(layers_sizes);
    DR_FREE(activation_functions);
    DR_FREE(activation_function_derivatives);
}

void dr_application_train_neural_network_current_data() {
    DR_ASSERT_MSG(training_current_dataset_index >= 0 && training_current_dataset_index < dataset_digits_count_all,
        "dataset index to train out of range the dataset in the application");

    DR_FLOAT_TYPE real_output[DR_APPLICATION_DIGITS_COUNT]     = { 0 };
    DR_FLOAT_TYPE expected_output[DR_APPLICATION_DIGITS_COUNT] = { 0 };
    DR_FLOAT_TYPE error[DR_APPLICATION_DIGITS_COUNT]           = { 0 };
    expected_output[dataset_digits_results[training_current_dataset_index]] = 1;
    dr_neural_network_set_input(neural_network,
        dataset_digits_pixels + training_current_dataset_index * DR_APPLICATION_CANVAS_PIXELS_COUNT);
    dr_neural_network_forward_propagation(neural_network);
    dr_neural_network_get_output(neural_network, real_output);
    for (size_t i = 0; i < DR_APPLICATION_DIGITS_COUNT; ++i) {
        error[i] = expected_output[i] - real_output[i];
    }
    dr_neural_network_back_propagation(neural_network, training_learning_rate, error);
}

void* dr_application_train_neural_network_other_thread(void*) {
    printf("start thread\n");
    while (training_process_active && training_current_epoch < training_epochs) {
        if (training_current_dataset_index >= dataset_digits_count_all) {
            training_current_dataset_index = 0;
            ++training_current_epoch;
        }
        dr_application_train_neural_network_current_data();
        ++training_current_dataset_index;
    }
    training_process_active   = false;
    training_procces_finished = true;
    training_current_dataset_index = 0;
    training_current_epoch = 0;
    printf("end thread\n");
    return NULL;
}

void dr_application_start_train_neural_network_other_thread() {
    const int result = pthread_create(
        &training_thread_id, NULL, dr_application_train_neural_network_other_thread, NULL);
    DR_ASSERT_MSG(result == 0, "error creating a thread for training a neural network in application");
}

void dr_application_training_tab_hidden_layers_list_view(const Rectangle list_view_bounds, const bool removed) {
    // list view status bar
    const Rectangle list_view_status_bar_bounds = {
        list_view_bounds.x,
        list_view_bounds.y - DR_APPLICATION_STATUS_BAR_HEIGHT,
        list_view_bounds.width,
        DR_APPLICATION_STATUS_BAR_HEIGHT
    };

    // gui
    const int training_list_view_active_new = GuiListViewEx(
        list_view_bounds, (const char**)training_hidden_layers_info, training_hidden_layers_count,
        &training_list_view_focus, &training_list_view_scroll_index, training_list_view_active);

    if (training_list_view_active_new != -1 &&
        ((training_list_view_active_new != training_list_view_active) || removed)) {
        char activation_function[DR_STR_BUFFER_SIZE];
        dr_application_training_get_hidden_layer(
            training_list_view_active_new, &training_controller_layer_size, activation_function);
        if (strcmp(activation_function, DR_APPLICATION_TRAINING_SIGMOID_STR) == 0) {
            training_controller_dropbox_index = 0;
        } else if (strcmp(activation_function, DR_APPLICATION_TRAINING_TANH_STR) == 0) {
            training_controller_dropbox_index = 1;
        }  else if (strcmp(activation_function, DR_APPLICATION_TRAINING_RELU_STR) == 0) {
            training_controller_dropbox_index = 2;
        } else {
            DR_ASSERT_MSG(false, "unknown activation function in application");
        }
    }
    training_list_view_active = training_list_view_active_new;

    GuiStatusBar(list_view_status_bar_bounds, "Hidden layers");
}

void dr_application_training_tab_hidden_layers_list_view_controllers(
    const Rectangle hidden_layers_bounds, const Rectangle list_view_bounds, const bool layer_selected, bool* removed) {
    // list view contoller container
    const Rectangle list_view_controller_bounds = {
        list_view_bounds.x,
        list_view_bounds.y + list_view_bounds.height,
        list_view_bounds.width,
        (hidden_layers_bounds.y + hidden_layers_bounds.height) - (list_view_bounds.y + list_view_bounds.height)
    };
    const Rectangle list_view_controller_toggle_group_bounds = {
        list_view_controller_bounds.x,
        list_view_controller_bounds.y,
        list_view_controller_bounds.width,
        list_view_controller_bounds.height / 3
    };

    // gui
    char toggle_text[DR_STR_BUFFER_SIZE] = { 0 };
    sprintf(toggle_text, "%s", "Add");
    if (layer_selected) {
        sprintf(toggle_text + TextLength(toggle_text), "%s", "\nRemove");
    }
    if (training_hidden_layers_count > 0) {
        sprintf(toggle_text + TextLength(toggle_text), "%s", "\nClear");
    }

    const int list_view_toggle_index = GuiToggleGroup(
        list_view_controller_toggle_group_bounds, toggle_text, -1);

    if (list_view_toggle_index <= -1) {
        return;
    }

    int split_toggle_text_count = 0;
    const char** split_toggle_text = TextSplit(toggle_text, '\n', &split_toggle_text_count);
    const char* clicked_toggle_text = split_toggle_text[list_view_toggle_index];
    if (strcmp(clicked_toggle_text, "Add") == 0) {
        dr_application_training_add_hidden_layer(
            DR_APPLICATION_CANVAS_PIXELS_COUNT, DR_APPLICATION_TRAINING_SIGMOID_STR);
    } else if (strcmp(clicked_toggle_text, "Remove") == 0) {
        dr_application_training_remove_hidden_layer(training_list_view_active);
        if (training_list_view_active >= training_hidden_layers_count) {
            training_list_view_active = training_hidden_layers_count - 1;
        }
        if (training_list_view_scroll_index > 0) {
            --training_list_view_scroll_index;
        }
        *removed = true;
    } else if (strcmp(clicked_toggle_text, "Clear") == 0) {
        dr_application_training_hidden_layers_info_clear();
        training_list_view_active = -1;
    }
}

void dr_application_training_tab_hidden_layers_list_view_item_controllers(const Rectangle hidden_layers_bounds,
    const Rectangle list_view_bounds, const bool layer_selected) {
    if (!layer_selected) {
        return;
    }

    // layer controller container
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

    // element
    const Vector2 layer_controller_element_size = {
        layer_controller_bounds.width - layer_controller_bounds_margin_h,
        layer_controller_bounds.height / 2
    };

    // spinner
    const Rectangle layer_controller_spinner_bounds = {
        layer_controller_bounds.x,
        layer_controller_bounds.y,
        layer_controller_element_size.x,
        layer_controller_element_size.y
    };
    const size_t training_controller_layer_size_last = training_controller_layer_size;

    // dropbox
    const Rectangle layer_controller_dropbox_bounds = {
        layer_controller_spinner_bounds.x,
        layer_controller_spinner_bounds.y + layer_controller_spinner_bounds.height,
        layer_controller_element_size.x,
        layer_controller_element_size.y
    };

    // gui
    if (GuiSpinner(layer_controller_spinner_bounds, NULL,
        (int*)&training_controller_layer_size, 1, INT_MAX, training_controller_layer_size_edit)) {
        training_controller_layer_size_edit = !training_controller_layer_size_edit;
    }

    const char* dropbox_text = DR_APPLICATION_TRAINING_SIGMOID_STR ";" DR_APPLICATION_TRAINING_TANH_STR ";"
        DR_APPLICATION_TRAINING_RELU_STR;
    const size_t training_controller_dropbox_index_last = training_controller_dropbox_index;
    if (GuiDropdownBox(layer_controller_dropbox_bounds, dropbox_text,
        &training_controller_dropbox_index, training_controller_dropbox_edit)) {
        training_controller_dropbox_edit = !training_controller_dropbox_edit;
    }

    if (training_controller_layer_size_last != training_controller_layer_size ||
        training_controller_dropbox_index_last != training_controller_dropbox_index) {
        int count = 0;
        const char** dropbox_split_text = TextSplit(dropbox_text, ';', &count);
        dr_application_training_set_hidden_layer(training_list_view_active, training_controller_layer_size,
            dropbox_split_text[training_controller_dropbox_index]);
    }

    return;
}

void dr_application_training_tab_hidden_layers(const Rectangle work_area) {
    const bool layer_selected = training_list_view_active >= 0;

    // hidden layers container
    const Vector2 hidden_layers_bounds_size = {
        work_area.width / 2,
        work_area.height / 2
    };
    const Rectangle hidden_layers_bounds = {
        work_area.x + work_area.width / 2 - hidden_layers_bounds_size.x / 2,
        work_area.y + work_area.height / 2.3 - hidden_layers_bounds_size.y / 2,
        hidden_layers_bounds_size.x,
        hidden_layers_bounds_size.y
    };

    // list view
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

    // preset
    const float dist_to_bottom =
        (work_area.y + work_area.height) - (hidden_layers_bounds.y + hidden_layers_bounds.height);

    const Vector2 train_preset_bounds_size = {
        hidden_layers_bounds.width / 2,
        dist_to_bottom / 2
    };
    const Rectangle train_preset_bounds = {
        hidden_layers_bounds.x + hidden_layers_bounds.width / 2 - hidden_layers_bounds.width / 4,
        (hidden_layers_bounds.y + hidden_layers_bounds.height) + dist_to_bottom / 2 - train_preset_bounds_size.y / 2,
        train_preset_bounds_size.x,
        train_preset_bounds_size.y
    };

    // preset element
    const Vector2 train_preset_element_size = {
        train_preset_bounds.width,
        train_preset_bounds.height / 4
    };

    // value box
    const Rectangle epochs_value_box_bounds = {
        train_preset_bounds.x + train_preset_bounds.width / 2 - train_preset_element_size.x / 2,
        train_preset_bounds.y + train_preset_bounds.height / 2 - train_preset_element_size.y,
        train_preset_element_size.x,
        train_preset_element_size.y
    };

    // button
    const Rectangle train_button_bounds = {
        epochs_value_box_bounds.x,
        epochs_value_box_bounds.y + epochs_value_box_bounds.height,
        train_preset_element_size.x,
        train_preset_element_size.y
    };

    // gui
    bool removed = false;
    dr_application_training_tab_hidden_layers_list_view_controllers(
        hidden_layers_bounds, list_view_bounds, layer_selected, &removed);
    dr_application_training_tab_hidden_layers_list_view(list_view_bounds, removed);
    dr_application_training_tab_hidden_layers_list_view_item_controllers(
        hidden_layers_bounds, list_view_bounds, layer_selected);

    if (GuiValueBox(
        epochs_value_box_bounds, "Epochs ", (int*)&training_epochs, 1, INT_MAX, training_epochs_value_box_edit)) {
        training_epochs_value_box_edit = !training_epochs_value_box_edit;
    }
    if (GuiButton(train_button_bounds, "Train")) {
        training_process_active = true;
        if (dr_neural_network_valid(neural_network)) {
            dr_neural_network_free(&neural_network);
        }
        dr_application_neural_network_create();
        dr_application_start_train_neural_network_other_thread();
    }
}

void dr_application_training_tab_training_process(const Rectangle work_area) {
    const Vector2 window_box_size = {
        work_area.width / 2.2,
        work_area.height / 2
    };
    const Rectangle window_box_bounds = {
        work_area.x + work_area.width / 2 - window_box_size.x / 2,
        work_area.y + work_area.height / 2 - window_box_size.y / 2,
        window_box_size.x,
        window_box_size.y
    };

    // window box
    const Vector2 window_box_content_size = {
        window_box_bounds.width / 1.5,
        window_box_bounds.height / 2
    };
    const Rectangle window_box_content_bounds = {
        window_box_bounds.x + window_box_bounds.width / 2 - window_box_content_size.x / 2,
        window_box_bounds.y + window_box_bounds.height / 2 - window_box_content_size.y / 2,
        window_box_content_size.x,
        window_box_content_size.y
    };

    // window box content elements
    const float window_box_content_elements_margin = 10;
    const Vector2 window_box_content_element_size = {
        window_box_content_bounds.width / 2,
        window_box_content_bounds.height / 8
    };

    // label error
    const Rectangle label_error_bounds = {
        window_box_content_bounds.x + window_box_content_bounds.width / 2 - window_box_content_element_size.x / 2,
        window_box_content_bounds.y + window_box_content_bounds.height / 2 -
            window_box_content_element_size.y / 2 - window_box_content_elements_margin / 2,
        window_box_content_element_size.x,
        window_box_content_element_size.y
    };

    // progress bar
    const Rectangle progress_bar_bounds = {
        window_box_bounds.x + window_box_bounds.width / 2 - window_box_content_element_size.x / 2,
        label_error_bounds.y + label_error_bounds.height + window_box_content_elements_margin / 2,
        window_box_content_element_size.x,
        window_box_content_element_size.y
    };

    // cancel button
    const float dist_to_bottom = window_box_bounds.y + window_box_bounds.height -
        (progress_bar_bounds.y + progress_bar_bounds.height);
    const Vector2 cancel_button_size = {
        window_box_bounds.width / 5,
        window_box_content_element_size.y
    };
    const Rectangle cancel_button_bounds = {
        window_box_bounds.x + window_box_bounds.width / 2 - cancel_button_size.x / 2,
        progress_bar_bounds.y + progress_bar_bounds.height + dist_to_bottom / 2 - cancel_button_size.y / 2,
        cancel_button_size.x,
        cancel_button_size.y
    };

    // gui
    const int window_box_res    = GuiWindowBox(window_box_bounds, "Neural network training process");
    const int cancel_button_res = GuiButton(cancel_button_bounds, "Cancel");
    if (cancel_button_res || window_box_res) {
        training_process_active = false;
        training_proccess_cancelled = true;
        return;
    }

    GuiLabel(label_error_bounds, TextFormat("%s: %.3f", "Error", training_error));

    training_current_epoch = GuiProgressBar(
        progress_bar_bounds, TextFormat("%zu ", training_current_epoch), TextFormat(" %zu", training_epochs),
        training_current_epoch, 0, training_epochs);
}

void dr_application_training_tab() {
    // work area
    const Rectangle work_area = {
        0,
        DR_APPLICATION_TAB_BOTTOM,
        window_size.x,
        window_size.y - DR_APPLICATION_TAB_BOTTOM
    };

    // message box cancelled
    const Vector2 message_box_cancelled_size = {
        work_area.width / 2,
        work_area.height / 4
    };
    const Rectangle message_box_cancelled_bounds = {
        work_area.x + work_area.width / 2 - message_box_cancelled_size.x / 2,
        work_area.y + work_area.height / 2 - message_box_cancelled_size.y / 2,
        message_box_cancelled_size.x,
        message_box_cancelled_size.y
    };

    // message box success
    const Vector2 message_box_success_size = {
        work_area.width / 2,
        work_area.height / 4
    };
    const Rectangle message_box_success_bounds = {
        work_area.x + work_area.width / 2 - message_box_success_size.x / 2,
        work_area.y + work_area.height / 2 - message_box_success_size.y / 2,
        message_box_success_size.x,
        message_box_success_size.y
    };

    // gui
    dr_application_training_tab_hidden_layers(work_area);

    if (training_process_active) {
        GuiUnlock();
        dr_gui_dim(work_area);
        dr_application_training_tab_training_process(work_area);
        GuiLock();
    } else if (training_proccess_cancelled) {
        GuiUnlock();
        dr_gui_dim(work_area);
        const char* message = TextFormat("Neural network training cancelled with an error: %.3f ", training_error);
        const int message_box_cancelled_result = GuiMessageBox(
            message_box_cancelled_bounds, "Cancelled", message, "Ok");
        if (message_box_cancelled_result == 0 || message_box_cancelled_result == 1) {
            training_proccess_cancelled = false;
            training_procces_finished   = false;
        } else {
            GuiLock();
        }
    } else if (training_procces_finished) {
        GuiUnlock();
        dr_gui_dim(work_area);
        const char* message = TextFormat(
            "The neural network has been successfully trained with an error: %.3f ", training_error);
        const int message_box_success_result = GuiMessageBox(message_box_success_bounds, "Success", message, "Ok");
        if (message_box_success_result >= 0) {
            training_procces_finished = false;
        } else {
            GuiLock();
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////// PREDICTION

void dr_application_prediction_tab() {
}

void dr_application_draw() {
    switch (current_tab) {
    case dr_application_tab_dataset:
        dr_application_dataset_tab();
        break;
    case dr_application_tab_training:
        dr_application_training_tab();
        break;
    case dr_application_tab_prediction:
        dr_application_prediction_tab();
        break;
    default:
        DR_ASSERT_MSG(false, "attempt to call the draw function for unknown application tab");
        break;
    }

    const Rectangle tab_rect = { 0, 0, window_size.x / DR_APPLICATION_TAB_COUNT, DR_APPLICATION_TAB_HEIGHT };
    current_tab = GuiToggleGroup(tab_rect, "Dataset;Trainig;Prediction", current_tab);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////// APPLICATION

void dr_application_start() {
    while (!WindowShouldClose()) {
        window_size = CLITERAL(Vector2){ (float)GetScreenWidth(), (float)GetScreenHeight() };

        BeginDrawing();
        ClearBackground(CLITERAL(Color){ 20, 20, 25, 255 });
        dr_application_draw();
        EndDrawing();
    }
}

void dr_application_create() {
    InitWindow(DR_APPLICATION_WINDOW_WIDTH, DR_APPLICATION_WINDOW_HEIGHT, DR_APPLICATION_DIGIT_RECOGNIZER_STR);
    SetTargetFPS(30);

    dataset_canvas_rtexture = LoadRenderTexture(
        DR_APPLICATION_CANVAS_RESOLUTION_WIDTH, DR_APPLICATION_CANVAS_RESOLUTION_HEIGHT);
    dr_application_dataset_canvas_clear();

    dr_application_training_add_hidden_layer(DR_APPLICATION_CANVAS_PIXELS_COUNT, DR_APPLICATION_TRAINING_SIGMOID_STR);
    for (size_t i = 2; i <= 8; i += 2) {
    dr_application_training_add_hidden_layer(
        DR_APPLICATION_CANVAS_PIXELS_COUNT / i, DR_APPLICATION_TRAINING_SIGMOID_STR);
    }
}

void dr_application_close() {
    // training
    if (training_process_active) {
        training_process_active = false;
        pthread_join(training_thread_id, NULL);
    }
    dr_application_training_hidden_layers_info_clear();
    if (dr_neural_network_valid(neural_network)) {
        dr_neural_network_free(&neural_network);
    }

    // dataset
    dr_application_dataset_clear();
    UnloadRenderTexture(dataset_canvas_rtexture);

    // raylib window
    CloseWindow();
}