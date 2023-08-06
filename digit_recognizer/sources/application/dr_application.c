#include <application/dr_application.h>
#include <application/dr_gui.h>
#include <application/dr_thread.h>
#include <neural_network/dr_neural_network.h>
#include <limits.h>

// #define DR_APPLICATION_SAVE_USER_NEURAL_NETWORK
#define DR_APPLICATION_SAVE_USER_NEURAL_NETWORK_PATH       "user_neural_network.txt"
#define DR_APPLICATION_LOAD_PRETRAINED_NEURAL_NETWORK_PATH "assets/pretrained_neural_network.txt"

#define DR_APPLICATION_WINDOW_WIDTH          800
#define DR_APPLICATION_WINDOW_HEIGHT         600
#define DR_APPLICATION_DIGIT_RECOGNIZER_STR  "Digit recognizer"
#define DR_APPLICATION_TAB_COUNT             3
#define DR_APPLICATION_TAB_HEIGHT            40
#define DR_APPLICATION_TAB_BOTTOM            DR_APPLICATION_TAB_HEIGHT
#define DR_APPLICATION_STATUS_BAR_HEIGHT     20
#define DR_APPLICATION_DIGITS_COUNT          10
#define DR_APPLICATION_TEXT_FORMAT_PRECISION "%.4f"

#define DR_APPLICATION_MNIST_DATASET_PATH         "assets/dataset.bin"
#define DR_APPLICATION_MNIST_DATASET_MAX_COUNT    60000
#define DR_APPLICATION_CANVAS_RESOLUTION_WIDTH    28
#define DR_APPLICATION_CANVAS_RESOLUTION_HEIGHT   28
#define DR_APPLICATION_CANVAS_PIXELS_COUNT        (DR_APPLICATION_CANVAS_RESOLUTION_WIDTH *\
    DR_APPLICATION_CANVAS_RESOLUTION_HEIGHT)
#define DR_APPLICATION_CANVAS_WIDTH               300
#define DR_APPLICATION_CANVAS_HEIGHT              300
#define DR_APPLICATION_CANVAS_DRAW_COLOR          CLITERAL(Color){ 255, 255, 255, 230 }
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
Vector2 window_size              = { DR_APPLICATION_WINDOW_WIDTH, DR_APPLICATION_WINDOW_HEIGHT };
dr_application_tab current_tab   = dr_application_tab_dataset;
dr_neural_network user_neural_network       = { 0 };
dr_neural_network pretrained_neural_network = { 0 };

// dataset
RenderTexture2D dataset_canvas_rtexture = { 0 };
Vector2 dataset_canvas_last_point       = { -1 };
char dataset_status_bar_str_buffer[DR_STR_BUFFER_SIZE]   = DR_APPLICATION_DIGIT_RECOGNIZER_STR;
size_t dataset_digits_count_total                        = 0;
size_t dataset_digits_count[DR_APPLICATION_DIGITS_COUNT] = { 0 };
DR_FLOAT_TYPE* dataset_digits_pixels                     = NULL;
unsigned char* dataset_digits_labels                     = NULL;
bool dataset_attempt_to_load_mnist        = false;
bool dataset_attempt_to_load_mnist_failed = false;
int dataset_count_to_load_mnist               = 100;
bool dataset_count_to_load_mnist_spinner_edit = false;

// training
int training_list_view_scroll_index = 0;
int training_list_view_active_index = -1;
int training_list_view_focus        = 0;
size_t training_controller_layer_size    = 1;
bool training_controller_layer_size_edit = false;
int training_controller_dropbox_index = 0;
bool training_controller_dropbox_edit = false;
size_t training_hidden_layers_count   = 0;
char** training_hidden_layers_info    = NULL;
DR_FLOAT_TYPE training_learning_rate  = 0.01;
size_t training_current_epoch = 0;
size_t training_count_epochs  = 1000;
bool training_epochs_spinner_edit = false;
bool training_attempt_to_start_training_failed = false;
bool training_process_active     = false;
bool training_procces_finished   = false;
bool training_proccess_stopped   = false;
bool training_neural_network_updated  = false;
DR_FLOAT_TYPE training_error          = 0;
size_t training_current_dataset_index = 0;
dr_thread_id_t training_thread_id         = { 0 };
dr_thread_handle_t training_thread_handle = 0;

// prediction
RenderTexture2D prediction_canvas_rtexture = { 0 };
Vector2 prediction_canvas_last_point       = { -1 };
bool prediction_show = false;
DR_FLOAT_TYPE prediction_probs[DR_APPLICATION_DIGITS_COUNT] = { 0 };
DR_FLOAT_TYPE prediction_max_prob = 0;
size_t prediction_predicted_digit = 0;
bool prediction_use_my_neural_network         = false;
bool prediction_use_pretrained_neural_network = false;
bool prediction_attempt_to_select_my_neural_network_failed         = false;
bool prediction_attempt_to_select_pretrained_neural_network_failed = false;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////// GENERAL

void dr_application_canvas_clear(RenderTexture2D canvas) {
    BeginTextureMode(canvas);
    ClearBackground(DR_APPLICATION_CANVAS_ERASE_COLOR);
    EndTextureMode();
}

void dr_application_canvas_get_pixels(const RenderTexture2D canvas, DR_FLOAT_TYPE* pixels) {
    Image canvas_image = LoadImageFromTexture(canvas.texture);
    for (size_t y = DR_APPLICATION_CANVAS_RESOLUTION_HEIGHT; y > 0; --y) {
        for (size_t x = 0; x < DR_APPLICATION_CANVAS_RESOLUTION_WIDTH; ++x) {
            const Color pixel_color = GetImageColor(canvas_image, x, y - 1);
            *pixels = (pixel_color.r + pixel_color.g + pixel_color.b) / (255.0f * 3.0f);
            ++pixels;
        }
    }
    UnloadImage(canvas_image);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////// DATASET

size_t dr_application_dataset_add_memory(const size_t count) {
    const size_t new_digits_count_total = dataset_digits_count_total + count;

    DR_FLOAT_TYPE* reallocated_pixels = (DR_FLOAT_TYPE*)DR_REALLOC(
        dataset_digits_pixels, sizeof(DR_FLOAT_TYPE) * new_digits_count_total * DR_APPLICATION_CANVAS_PIXELS_COUNT);
    DR_ASSERT_MSG(reallocated_pixels, "new pixels reallocate error for the application dataset");

    unsigned char* reallocated_results = (unsigned char*)DR_REALLOC(
        dataset_digits_labels, sizeof(unsigned char) * new_digits_count_total);
    DR_ASSERT_MSG(reallocated_results, "results reallocate error for the application dataset");

    dataset_digits_pixels  = reallocated_pixels;
    dataset_digits_labels = reallocated_results;

    return new_digits_count_total;
}

void dr_application_dataset_add_digit(const unsigned char digit) {
    DR_ASSERT_MSG(digit >= 0 && digit <= 9, "attempt to add a not correct digit to the application dataset");

    const size_t new_digits_count_total = dr_application_dataset_add_memory(1);
    DR_FLOAT_TYPE* new_pixels = dataset_digits_pixels + dataset_digits_count_total * DR_APPLICATION_CANVAS_PIXELS_COUNT;
    dr_application_canvas_get_pixels(dataset_canvas_rtexture, new_pixels);
    dataset_digits_count_total = new_digits_count_total;
    dataset_digits_labels[dataset_digits_count_total - 1] = digit;
    ++dataset_digits_count[digit];
}

bool dr_application_dataset_load_mnist(const size_t count) {
    FILE* file = fopen(DR_APPLICATION_MNIST_DATASET_PATH, "rb");
    if (!file) {
        return false;
    }

    uint32_t header[3] = { 0 };
    fread(header, sizeof(header), 1, file);

    if (count > header[0]) {
        dr_print_error("Attempt to load more MNIST images than there are");
        fclose(file);
        return false;
    }

    if (header[1] != DR_APPLICATION_CANVAS_RESOLUTION_HEIGHT ||
        header[2] != DR_APPLICATION_CANVAS_RESOLUTION_WIDTH) {
        dr_print_error("Mismatch application canvas resolution and MNSIT images resolution\n");
        fclose(file);
        return false;
    }

    const size_t new_digits_count_total = dr_application_dataset_add_memory(count);
    DR_FLOAT_TYPE* new_pixels = dataset_digits_pixels + dataset_digits_count_total * DR_APPLICATION_CANVAS_PIXELS_COUNT;
    unsigned char* new_results = dataset_digits_labels + dataset_digits_count_total; 

    const size_t pixels_count_total = count * DR_APPLICATION_CANVAS_PIXELS_COUNT;
    const DR_FLOAT_TYPE* new_pixels_end = new_pixels + pixels_count_total;
    for (; new_pixels != new_pixels_end; ++new_pixels) {
        unsigned char pixel = 0;
        fread(&pixel, sizeof(unsigned char), 1, file);
        *new_pixels = (float)pixel / 255.0f;
    }

    fseek(file, sizeof(header) + sizeof(unsigned char) * header[0] * DR_APPLICATION_CANVAS_PIXELS_COUNT, SEEK_SET);

    const unsigned char* new_results_end = new_results + count;
    for (; new_results != new_results_end; ++new_results) {
        unsigned char digit = 0;
        fread(&digit, sizeof(unsigned char), 1, file);
        *new_results = digit;
        ++dataset_digits_count[digit];
    }

    dataset_digits_count_total = new_digits_count_total;

    fclose(file);

    return true;
}

void dr_application_dataset_clear() {
    if (dataset_digits_count_total == 0) {
        return;
    }

    for (size_t i = 0; i < DR_APPLICATION_DIGITS_COUNT; ++i) {
        dataset_digits_count[i] = 0;
    }

    DR_FREE(dataset_digits_pixels);
    dataset_digits_pixels = NULL;
    DR_FREE(dataset_digits_labels);
    dataset_digits_labels = NULL;
    dataset_digits_count_total = 0;
}

void dr_application_dataset_tab() {
    const bool dataset_empty = dataset_digits_count_total == 0;

    // work area
    Rectangle work_area = { 0 };
    work_area.x = 0;
    work_area.y = DR_APPLICATION_TAB_BOTTOM;
    work_area.width  = window_size.x;
    work_area.height = window_size.y - DR_APPLICATION_STATUS_BAR_HEIGHT - DR_APPLICATION_TAB_BOTTOM;

    // dim area
    Rectangle dim_area = { 0 };
    dim_area.x = 0;
    dim_area.y = DR_APPLICATION_TAB_BOTTOM;
    dim_area.width  = window_size.x;
    dim_area.height = window_size.y - DR_APPLICATION_TAB_BOTTOM;

    // canvas
    Rectangle canvas_bounds = { 0 };
    canvas_bounds.x = work_area.x + (work_area.width / 2 - DR_APPLICATION_CANVAS_WIDTH / 2);
    canvas_bounds.y = work_area.y +
        (work_area.height / 2.2 - DR_APPLICATION_CANVAS_HEIGHT / 2) - DR_APPLICATION_CANVAS_CLEAR_BUTTON_HEIGHT;
    canvas_bounds.width  = DR_APPLICATION_CANVAS_WIDTH;
    canvas_bounds.height = DR_APPLICATION_CANVAS_HEIGHT;

    // numeric buttons
    Rectangle numeric_buttons_bounds = { 0 };
    numeric_buttons_bounds.x = work_area.x;
    numeric_buttons_bounds.y = canvas_bounds.y + DR_APPLICATION_CANVAS_WINDOW_HEIGHT;
    numeric_buttons_bounds.width  = work_area.width;
    numeric_buttons_bounds.height = work_area.height - (canvas_bounds.y + canvas_bounds.height);

    // window box bounds
    Rectangle window_box_bounds = { 0 };
    window_box_bounds.width  = work_area.width / 2.2;
    window_box_bounds.height = work_area.height / 2;
    window_box_bounds.x = work_area.x + work_area.width / 2 - window_box_bounds.width / 2;
    window_box_bounds.y = work_area.y + work_area.height / 2 - window_box_bounds.height / 2;

    // window box
    const char* window_box_text = "How many images do you want to load?";
    const Font window_box_font  = GetFontDefault();
    const int window_box_font_size    = GuiGetStyle(DEFAULT, TEXT_SIZE);
    const int window_box_text_spacing = GuiGetStyle(DEFAULT, TEXT_SPACING);
    const Color window_box_text_color = GetColor(GuiGetStyle(DEFAULT, TEXT_COLOR_NORMAL));

    const Vector2 window_box_text_size = MeasureTextEx(
        window_box_font, window_box_text, window_box_font_size, window_box_text_spacing);

    Vector2 window_box_text_pos = { 0 };
    window_box_text_pos.x = window_box_bounds.x + window_box_bounds.width / 2 - window_box_text_size.x / 2;
    window_box_text_pos.y = window_box_bounds.y + window_box_bounds.height / 3 - window_box_text_size.y / 2;

    // window box spinner
    Rectangle window_box_spinner_bounds = { 0 };
    window_box_spinner_bounds.width  = window_box_bounds.width / 2;
    window_box_spinner_bounds.height = 20;
    window_box_spinner_bounds.x =
        window_box_bounds.x + window_box_bounds.width / 2 - window_box_spinner_bounds.width / 2;
    window_box_spinner_bounds.y = window_box_text_pos.y + window_box_text_size.y + window_box_bounds.height / 5;

    // window box toggle group
    Rectangle window_box_toggle_group_bounds = { 0 };
    window_box_toggle_group_bounds.width  = window_box_bounds.width / 4;
    window_box_toggle_group_bounds.height = 20;
    window_box_toggle_group_bounds.x =
        window_box_bounds.x + window_box_bounds.width / 2 - window_box_toggle_group_bounds.width;
    window_box_toggle_group_bounds.y =
        window_box_spinner_bounds.y + window_box_spinner_bounds.height + window_box_bounds.height / 5;

    // message box
    Rectangle message_box_bounds = { 0 };
    message_box_bounds.width  = work_area.width / 1.5;
    message_box_bounds.height = work_area.height / 3;
    message_box_bounds.x = work_area.x + work_area.width / 2 - message_box_bounds.width / 2;
    message_box_bounds.y = work_area.y + work_area.height / 2 - message_box_bounds.height / 2;

    // status bar
    Rectangle status_bar_bounds = { 0 };
    status_bar_bounds.x = 0;
    status_bar_bounds.y = window_size.y - DR_APPLICATION_STATUS_BAR_HEIGHT;
    status_bar_bounds.width  = window_size.x;
    status_bar_bounds.height = DR_APPLICATION_STATUS_BAR_HEIGHT;

    Vector2 status_bar_buttons_size = { 0 };
    status_bar_buttons_size.x = status_bar_bounds.width / 5;
    status_bar_buttons_size.y = status_bar_bounds.height;

    // load MNIST button
    Rectangle load_mnist_button_bounds = { 0 };
    load_mnist_button_bounds.x = status_bar_bounds.x + status_bar_bounds.width - status_bar_buttons_size.x;
    load_mnist_button_bounds.y = status_bar_bounds.y;
    load_mnist_button_bounds.width  = status_bar_buttons_size.x;
    load_mnist_button_bounds.height = status_bar_buttons_size.y;

    // clear dataset button
    Rectangle clear_dataset_button_bounds = { 0 };
    clear_dataset_button_bounds.x = load_mnist_button_bounds.x - status_bar_buttons_size.x;
    clear_dataset_button_bounds.y = status_bar_bounds.y;
    clear_dataset_button_bounds.width  = status_bar_buttons_size.x;
    clear_dataset_button_bounds.height = status_bar_buttons_size.y;

    // gui
    dataset_canvas_last_point = dr_gui_canvas(
        canvas_bounds, DR_APPLICATION_CANVAS_CLEAR_BUTTON_HEIGHT, dataset_canvas_rtexture,
        DR_APPLICATION_CANVAS_DRAW_COLOR, DR_APPLICATION_CANVAS_ERASE_COLOR, dataset_canvas_last_point);

    const int clicked_digit = dr_gui_numeric_buttons_row(
        numeric_buttons_bounds, DR_APPLICATION_DIGITS_COUNT, dataset_digits_count);
    if (clicked_digit >= 0) {
        dr_application_dataset_add_digit(clicked_digit);
        dr_application_canvas_clear(dataset_canvas_rtexture);
        sprintf(dataset_status_bar_str_buffer, "%s %d %s", "Digit", clicked_digit, "was added");
    }

    GuiStatusBar(status_bar_bounds, dataset_status_bar_str_buffer);

    if (!dataset_empty && GuiButton(clear_dataset_button_bounds, "Clear dataset")) {
        dr_application_dataset_clear();
        sprintf(dataset_status_bar_str_buffer, "%s", "Dataset has been cleared");
    }
    if (GuiButton(load_mnist_button_bounds, "Load MNIST")) {
        dataset_attempt_to_load_mnist = true;
    }

    if (dataset_attempt_to_load_mnist) {
        GuiUnlock();
        dr_gui_dim(dim_area);

        const int window_box_result = GuiWindowBox(window_box_bounds, "Loading the MNIST dataset");

        DrawTextEx(window_box_font, window_box_text,
            window_box_text_pos, window_box_font_size, window_box_text_spacing, window_box_text_color);

        const bool window_box_spinner_edit_changed = GuiSpinner(
            window_box_spinner_bounds, "Count ", &dataset_count_to_load_mnist,
            1, DR_APPLICATION_MNIST_DATASET_MAX_COUNT, dataset_count_to_load_mnist_spinner_edit);
        if (window_box_spinner_edit_changed) {
            dataset_count_to_load_mnist_spinner_edit = !dataset_count_to_load_mnist_spinner_edit;
        }

        const int window_box_toggle_index = GuiToggleGroup(window_box_toggle_group_bounds, "Load;Cancel", -1);
        if (window_box_toggle_index == 0) {
            if (dr_application_dataset_load_mnist(dataset_count_to_load_mnist)) {
                sprintf(dataset_status_bar_str_buffer, "%s(%d) %s",
                    "MNIST dataset", dataset_count_to_load_mnist, "has been loaded");
            } else {
                dataset_attempt_to_load_mnist_failed = true;
            }
            dataset_attempt_to_load_mnist = false;
            return;
        }

        if (window_box_result || window_box_toggle_index == 1) {
            dataset_attempt_to_load_mnist = false;
        } else {
            GuiLock();
        }
    } else if (dataset_attempt_to_load_mnist_failed) {
        GuiUnlock();
        dr_gui_dim(dim_area);

        const int message_box_result = GuiMessageBox(message_box_bounds,
            "Loading MNIST dataset failed",
            "An error occurred while trying to load MNIST dataset",
            "Ok");
        if (message_box_result >= 0) {
            dataset_attempt_to_load_mnist_failed = false;
        } else {
            GuiLock();
        }
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
    char** new_hidden_layers_info = (char**)DR_MALLOC(sizeof(char*) * new_layers_count);
    DR_ASSERT_MSG(new_hidden_layers_info, "apllication hidden layer remove - new hidden layers info alloc error");

    size_t new_arr_i = 0;
    for (size_t i = 0; i < training_hidden_layers_count; ++i) {
        if (i == index) {
            DR_FREE(training_hidden_layers_info[i]);
            continue;
        }
        new_hidden_layers_info[new_arr_i] = (char*)DR_MALLOC(sizeof(char) * DR_STR_BUFFER_SIZE);
        DR_ASSERT_MSG(new_hidden_layers_info[new_arr_i],
            "apllication hidden layer remove - new hidden layer info alloc error");
        memcpy(new_hidden_layers_info[new_arr_i], training_hidden_layers_info[i], DR_STR_BUFFER_SIZE);
        DR_FREE(training_hidden_layers_info[i]);
        ++new_arr_i;
    }
    DR_FREE(training_hidden_layers_info);

    training_hidden_layers_info  = new_hidden_layers_info;
    training_hidden_layers_count = new_layers_count;
}

void dr_application_training_set_hidden_layer(
    const size_t layer_index, const size_t layer_size, const char* activation_function) {
    DR_ASSERT_MSG(layer_index < training_hidden_layers_count,
        "index out of range when trying to set a hidden layer in application");

    const char* text_info = TextFormat("%zu %s", layer_size, activation_function);
    TextCopy(training_hidden_layers_info[layer_index], text_info);
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

    layers_sizes[0]                = DR_APPLICATION_CANVAS_PIXELS_COUNT;
    layers_sizes[layers_count - 1] = DR_APPLICATION_DIGITS_COUNT;
    activation_functions[activation_functions_count - 1]            = dr_sigmoid;
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

    user_neural_network = dr_neural_network_create(
        layers_sizes, layers_count, activation_functions, activation_function_derivatives);
    dr_neural_network_randomize_weights(user_neural_network, -1, 1);

    DR_FREE(layers_sizes);
    DR_FREE(activation_functions);
    DR_FREE(activation_function_derivatives);
}

void dr_application_train_neural_network_current_data() {
    DR_ASSERT_MSG(training_current_dataset_index >= 0 && training_current_dataset_index < dataset_digits_count_total,
        "dataset index to train out of range the dataset in the application");

    DR_FLOAT_TYPE real_output[DR_APPLICATION_DIGITS_COUNT]     = { 0 };
    DR_FLOAT_TYPE expected_output[DR_APPLICATION_DIGITS_COUNT] = { 0 };
    DR_FLOAT_TYPE error_output[DR_APPLICATION_DIGITS_COUNT]    = { 0 };
    expected_output[dataset_digits_labels[training_current_dataset_index]] = 1;
    dr_neural_network_set_input(user_neural_network,
        dataset_digits_pixels + training_current_dataset_index * DR_APPLICATION_CANVAS_PIXELS_COUNT);
    dr_neural_network_forward_propagation(user_neural_network);
    dr_neural_network_get_output(user_neural_network, real_output);
    DR_FLOAT_TYPE error_sum = 0;
    for (size_t i = 0; i < DR_APPLICATION_DIGITS_COUNT; ++i) {
        error_output[i] = expected_output[i] - real_output[i];
        error_sum += error_output[i];
    }
    training_error = fabs(error_sum);
    dr_neural_network_back_propagation(user_neural_network, training_learning_rate, error_output);
}

dr_thread_function_result_t DR_WINAPI dr_application_train_neural_network_other_thread(void* data) {
    training_error = 0;
    while (training_process_active && training_current_epoch < training_count_epochs) {
        if (training_current_dataset_index >= dataset_digits_count_total) {
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

#ifdef DR_APPLICATION_SAVE_USER_NEURAL_NETWORK
    if (!dr_neural_network_save_to_file(user_neural_network, DR_APPLICATION_SAVE_USER_NEURAL_NETWORK_PATH)) {
        dr_print_error("Error saving the user neural network");
    }
#endif // DR_APPLICATION_SAVE_USER_NEURAL_NETOWRK

    return 0;
}

void dr_application_start_train_neural_network_other_thread() {
    training_thread_handle = dr_thread_create(&training_thread_id, dr_application_train_neural_network_other_thread);
    DR_ASSERT_MSG(dr_check_thread_handle(training_thread_handle),
        "error creating a thread for training a neural network in application");
}

void dr_application_training_tab_hidden_layers_list_view(const Rectangle list_view_bounds, const bool removed) {
    // list view title
    Rectangle list_view_title_bounds = { 0 };
    list_view_title_bounds.x = list_view_bounds.x;
    list_view_title_bounds.y = list_view_bounds.y - DR_APPLICATION_STATUS_BAR_HEIGHT;
    list_view_title_bounds.width  = list_view_bounds.width;
    list_view_title_bounds.height = DR_APPLICATION_STATUS_BAR_HEIGHT;

    // gui
    const int training_list_view_active_index_new = GuiListViewEx(
        list_view_bounds, (const char**)training_hidden_layers_info, training_hidden_layers_count,
        &training_list_view_focus, &training_list_view_scroll_index, training_list_view_active_index);

    if (training_list_view_active_index_new != -1 &&
        ((training_list_view_active_index_new != training_list_view_active_index) || removed)) {
        char activation_function[DR_STR_BUFFER_SIZE];
        dr_application_training_get_hidden_layer(
            training_list_view_active_index_new, &training_controller_layer_size, activation_function);
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
    training_list_view_active_index = training_list_view_active_index_new;

    GuiStatusBar(list_view_title_bounds, "Hidden layers");
}

void dr_application_training_tab_hidden_layers_list_view_controllers(
    const Rectangle hidden_layers_bounds, const Rectangle list_view_bounds, const bool layer_selected, bool* removed) {
    // list view contoller container
    Rectangle list_view_controller_bounds = { 0 };
    list_view_controller_bounds.x = list_view_bounds.x;
    list_view_controller_bounds.y = list_view_bounds.y + list_view_bounds.height;
    list_view_controller_bounds.width  = list_view_bounds.width;
    list_view_controller_bounds.height =
        (hidden_layers_bounds.y + hidden_layers_bounds.height) - (list_view_bounds.y + list_view_bounds.height);

    // list view controller toggle group bounds
    Rectangle list_view_controller_toggle_group_bounds = { 0 };
    list_view_controller_toggle_group_bounds.x = list_view_controller_bounds.x;
    list_view_controller_toggle_group_bounds.y = list_view_controller_bounds.y;
    list_view_controller_toggle_group_bounds.width  = list_view_controller_bounds.width;
    list_view_controller_toggle_group_bounds.height = list_view_controller_bounds.height / 3;

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
        training_neural_network_updated = true;
    } else if (strcmp(clicked_toggle_text, "Remove") == 0) {
        dr_application_training_remove_hidden_layer(training_list_view_active_index);
        if (training_list_view_active_index >= training_hidden_layers_count) {
            training_list_view_active_index = training_hidden_layers_count - 1;
        }
        if (training_list_view_scroll_index > 0) {
            --training_list_view_scroll_index;
        }
        *removed = true;
        training_neural_network_updated = true;
    } else if (strcmp(clicked_toggle_text, "Clear") == 0) {
        dr_application_training_hidden_layers_info_clear();
        training_list_view_active_index = -1;
        training_neural_network_updated = true;
    }
}

void dr_application_training_tab_hidden_layers_list_view_item_controllers(
    const Rectangle hidden_layers_bounds, const Rectangle list_view_bounds, const bool layer_selected) {
    if (!layer_selected) {
        return;
    }

    // layer controller container
    const float layer_controller_bounds_margin_h = 5;
    Rectangle layer_controller_bounds = { 0 };
    layer_controller_bounds.width  =
        hidden_layers_bounds.width - list_view_bounds.width - layer_controller_bounds_margin_h;
    layer_controller_bounds.height = hidden_layers_bounds.height / 6;
    layer_controller_bounds.x = hidden_layers_bounds.x + hidden_layers_bounds.width -
        layer_controller_bounds.width + layer_controller_bounds_margin_h;
    layer_controller_bounds.y =
        hidden_layers_bounds.y + hidden_layers_bounds.height / 2 - layer_controller_bounds.height / 2;

    // layer controller element
    Vector2 layer_controller_element_size = { 0 };
    layer_controller_element_size.x = layer_controller_bounds.width - layer_controller_bounds_margin_h;
    layer_controller_element_size.y = layer_controller_bounds.height / 2;

    // layer controller spinner
    Rectangle layer_controller_spinner_bounds = { 0 };
    layer_controller_spinner_bounds.x = layer_controller_bounds.x;
    layer_controller_spinner_bounds.y = layer_controller_bounds.y;
    layer_controller_spinner_bounds.width  = layer_controller_element_size.x;
    layer_controller_spinner_bounds.height = layer_controller_element_size.y;
    const size_t training_controller_layer_size_old = training_controller_layer_size;

    // layer controller dropbox
    Rectangle layer_controller_dropbox_bounds = { 0 };
    layer_controller_dropbox_bounds.x = layer_controller_spinner_bounds.x;
    layer_controller_dropbox_bounds.y = layer_controller_spinner_bounds.y + layer_controller_spinner_bounds.height;
    layer_controller_dropbox_bounds.width  = layer_controller_element_size.x;
    layer_controller_dropbox_bounds.height = layer_controller_element_size.y;

    // gui
    if (GuiSpinner(layer_controller_spinner_bounds, NULL,
        (int*)&training_controller_layer_size, 1, INT_MAX, training_controller_layer_size_edit)) {
        training_controller_layer_size_edit = !training_controller_layer_size_edit;
    }

    const char* dropbox_text = 
        DR_APPLICATION_TRAINING_SIGMOID_STR ";" DR_APPLICATION_TRAINING_TANH_STR ";" DR_APPLICATION_TRAINING_RELU_STR;
    const size_t training_controller_dropbox_index_last = training_controller_dropbox_index;
    if (GuiDropdownBox(layer_controller_dropbox_bounds, dropbox_text,
        &training_controller_dropbox_index, training_controller_dropbox_edit)) {
        training_controller_dropbox_edit = !training_controller_dropbox_edit;
    }

    if (training_controller_layer_size_old != training_controller_layer_size ||
        training_controller_dropbox_index_last != training_controller_dropbox_index) {
        int dropbox_split_text_count = 0;
        const char** dropbox_split_text = TextSplit(dropbox_text, ';', &dropbox_split_text_count);
        dr_application_training_set_hidden_layer(training_list_view_active_index, training_controller_layer_size,
            dropbox_split_text[training_controller_dropbox_index]);
        training_neural_network_updated = true;
    }
}

void dr_application_training_tab_hidden_layers(const Rectangle work_area) {
    const bool layer_selected = training_list_view_active_index >= 0;

    // hidden layers container
    Rectangle hidden_layers_bounds = { 0 };
    hidden_layers_bounds.width = work_area.width / 2;
    hidden_layers_bounds.height = work_area.height / 2;
    hidden_layers_bounds.x = work_area.x + work_area.width / 2 - hidden_layers_bounds.width / 2;
    hidden_layers_bounds.y = work_area.y + work_area.height / 2.3 - hidden_layers_bounds.height / 2;

    // list view
    Rectangle list_view_bounds = { 0 };
    list_view_bounds.width  = hidden_layers_bounds.width / 2;
    list_view_bounds.height = hidden_layers_bounds.height / 1.3;
    list_view_bounds.x =
        hidden_layers_bounds.x + (hidden_layers_bounds.width / 2 - list_view_bounds.width / 2) * !layer_selected;
    list_view_bounds.y = hidden_layers_bounds.y;

    // settings
    const float dist_to_bottom =
        (work_area.y + work_area.height) - (hidden_layers_bounds.y + hidden_layers_bounds.height);

    Rectangle train_settings_bounds = { 0 };
    train_settings_bounds.width  = hidden_layers_bounds.width / 2;
    train_settings_bounds.height = dist_to_bottom / 2;
    train_settings_bounds.x = hidden_layers_bounds.x + hidden_layers_bounds.width / 2 - hidden_layers_bounds.width / 4;
    train_settings_bounds.y = (hidden_layers_bounds.y + hidden_layers_bounds.height) +
        dist_to_bottom / 2 - train_settings_bounds.height / 2;

    // settings element
    Vector2 train_settings_element_size = { 0 };
    train_settings_element_size.x = train_settings_bounds.width;
    train_settings_element_size.y = train_settings_bounds.height / 4;
    const float train_settings_height = train_settings_element_size.y * 3;

    // slider learning rate
    Rectangle learning_rate_slider_bounds = { 0 };
    learning_rate_slider_bounds.x =
        train_settings_bounds.x + train_settings_bounds.width / 2 - train_settings_element_size.x / 2;
    learning_rate_slider_bounds.y =
        train_settings_bounds.y + train_settings_bounds.height / 2 - train_settings_height / 2;
    learning_rate_slider_bounds.width  = train_settings_element_size.x;
    learning_rate_slider_bounds.height = train_settings_element_size.y;

    // value box epochs
    Rectangle epochs_value_box_bounds = { 0 };
    epochs_value_box_bounds.x = learning_rate_slider_bounds.x;
    epochs_value_box_bounds.y = learning_rate_slider_bounds.y + learning_rate_slider_bounds.height;
    epochs_value_box_bounds.width  = train_settings_element_size.x;
    epochs_value_box_bounds.height = train_settings_element_size.y;

    // button
    Rectangle train_button_bounds = { 0 };
    train_button_bounds.x = epochs_value_box_bounds.x;
    train_button_bounds.y = epochs_value_box_bounds.y + epochs_value_box_bounds.height;
    train_button_bounds.width  = train_settings_element_size.x;
    train_button_bounds.height = train_settings_element_size.y;

    // gui
    bool removed = false;
    dr_application_training_tab_hidden_layers_list_view_controllers(
        hidden_layers_bounds, list_view_bounds, layer_selected, &removed);
    dr_application_training_tab_hidden_layers_list_view(list_view_bounds, removed);
    dr_application_training_tab_hidden_layers_list_view_item_controllers(
        hidden_layers_bounds, list_view_bounds, layer_selected);

    const float min_learning_rate = 0.01;
    const float max_learning_rate = 0.1;
    char slider_left_text[DR_STR_BUFFER_SIZE] = { 0 };
    TextCopy(slider_left_text, TextFormat(
        "%s: "DR_APPLICATION_TEXT_FORMAT_PRECISION, "Learning rate", training_learning_rate));
    const char* slider_right_text = TextFormat(DR_APPLICATION_TEXT_FORMAT_PRECISION, max_learning_rate);
    training_learning_rate = GuiSlider(learning_rate_slider_bounds, slider_left_text, slider_right_text,
        training_learning_rate, min_learning_rate, max_learning_rate);

    if (GuiSpinner(epochs_value_box_bounds, "Epochs ",
        (int*)&training_count_epochs, 1, INT_MAX, training_epochs_spinner_edit)) {
        training_epochs_spinner_edit = !training_epochs_spinner_edit;
    }
    if (GuiButton(train_button_bounds, "Train")) {
        if (dataset_digits_count_total == 0) {
            training_attempt_to_start_training_failed = true;
            return;
        }

        training_process_active = true;
        if (user_neural_network.layers_count == 0) {
            dr_application_neural_network_create();
        } else if (training_neural_network_updated) {
            dr_neural_network_free(&user_neural_network);
            dr_application_neural_network_create();
            training_neural_network_updated = false;
        }
        dr_application_start_train_neural_network_other_thread();
    }
}

void dr_application_training_tab_training_process(const Rectangle work_area) {
    // window box
    Rectangle window_box_bounds = { 0 };
    window_box_bounds.width  = work_area.width / 2.2;
    window_box_bounds.height = work_area.height / 2;
    window_box_bounds.x = work_area.x + work_area.width / 2 - window_box_bounds.width / 2;
    window_box_bounds.y = work_area.y + work_area.height / 2 - window_box_bounds.height / 2;

    // window box content
    Rectangle window_box_content_bounds = { 0 };
    window_box_content_bounds.width  = window_box_bounds.width / 1.5;
    window_box_content_bounds.height = window_box_bounds.height / 2;
    window_box_content_bounds.x =
        window_box_bounds.x + window_box_bounds.width / 2 - window_box_content_bounds.width / 2;
    window_box_content_bounds.y =
        window_box_bounds.y + window_box_bounds.height / 2 - window_box_content_bounds.height / 2;

    // window box content elements
    const float window_box_content_elements_margin = 10;
    Vector2 window_box_content_element_size = { 0 };
    window_box_content_element_size.x = window_box_content_bounds.width / 2;
    window_box_content_element_size.y = window_box_content_bounds.height / 8;

    // label error
    Rectangle label_error_bounds = { 0 };
    label_error_bounds.x =
        window_box_content_bounds.x + window_box_content_bounds.width / 2 - window_box_content_element_size.x / 2;
    label_error_bounds.y = window_box_content_bounds.y + window_box_content_bounds.height / 2 -
            window_box_content_element_size.y / 2 - window_box_content_elements_margin / 2;
    label_error_bounds.width  = window_box_content_element_size.x;
    label_error_bounds.height = window_box_content_element_size.y;

    // progress bar
    Rectangle progress_bar_bounds = { 0 };
    progress_bar_bounds.x = window_box_bounds.x + window_box_bounds.width / 2 - window_box_content_element_size.x / 2;
    progress_bar_bounds.y = label_error_bounds.y + label_error_bounds.height + window_box_content_elements_margin / 2;
    progress_bar_bounds.width  = window_box_content_element_size.x;
    progress_bar_bounds.height = window_box_content_element_size.y;

    // stop button
    const float dist_from_progress_bar_to_bottom =
        window_box_bounds.y + window_box_bounds.height - (progress_bar_bounds.y + progress_bar_bounds.height);
    Vector2 stop_button_size = { 0 };
    stop_button_size.x = window_box_bounds.width / 5;
    stop_button_size.y = window_box_content_element_size.y;

    Rectangle stop_button_bounds = { 0 };
    stop_button_bounds.x = window_box_bounds.x + window_box_bounds.width / 2 - stop_button_size.x / 2;
    stop_button_bounds.y = progress_bar_bounds.y + progress_bar_bounds.height +
        dist_from_progress_bar_to_bottom / 2 - stop_button_size.y / 2;
    stop_button_bounds.width  = stop_button_size.x;
    stop_button_bounds.height = stop_button_size.y;

    // gui
    const int window_box_res  = GuiWindowBox(window_box_bounds, "Neural network training process");
    const int stop_button_res = GuiButton(stop_button_bounds, "Stop");
    if (stop_button_res || window_box_res) {
        training_process_active = false;
        const bool thread_join_result = dr_thread_join(training_thread_handle, training_thread_id);
        DR_ASSERT_MSG(thread_join_result, "thread join error when training the neural network in the application");
        training_proccess_stopped = true;
        return;
    }

    GuiLabel(label_error_bounds, TextFormat("%s: "DR_APPLICATION_TEXT_FORMAT_PRECISION, "Error", training_error));

    training_current_epoch = GuiProgressBar(
        progress_bar_bounds, TextFormat("%zu ", training_current_epoch), TextFormat(" %zu", training_count_epochs),
        training_current_epoch, 0, training_count_epochs);
}

void dr_application_training_tab() {
    // work area
    Rectangle work_area = { 0 };
    work_area.x = 0;
    work_area.y = DR_APPLICATION_TAB_BOTTOM;
    work_area.width  = window_size.x;
    work_area.height = window_size.y - DR_APPLICATION_TAB_BOTTOM;

    // message box
    Rectangle message_box_bounds = { 0 };
    message_box_bounds.width  = work_area.width / 1.6;
    message_box_bounds.height = work_area.height / 4;
    message_box_bounds.x = work_area.x + work_area.width / 2 - message_box_bounds.width / 2;
    message_box_bounds.y = work_area.y + work_area.height / 2 - message_box_bounds.height / 2;

    // gui
    dr_application_training_tab_hidden_layers(work_area);

    if (training_attempt_to_start_training_failed) {
        GuiUnlock();
        dr_gui_dim(work_area);
        const char* message = "To train the neural network, add an image of digit to the dataset.";
        const int message_box_result = GuiMessageBox(message_box_bounds, "Empty dataset", message, "Ok");
        if (message_box_result == 0 || message_box_result == 1) {
            training_attempt_to_start_training_failed = false;
        } else {
            GuiLock();
        }
        return;
    }
    
    if (training_process_active) {
        GuiUnlock();
        dr_gui_dim(work_area);
        dr_application_training_tab_training_process(work_area);
        GuiLock();
    } else if (training_proccess_stopped) {
        GuiUnlock();
        dr_gui_dim(work_area);
        const char* message = TextFormat(
            "Neural network training stopped with a last error: "DR_APPLICATION_TEXT_FORMAT_PRECISION, training_error);
        const int message_box_result = GuiMessageBox(message_box_bounds, "Stopped", message, "Ok");
        if (message_box_result == 0 || message_box_result == 1) {
            training_proccess_stopped = false;
            training_procces_finished   = false;
        } else {
            GuiLock();
        }
    } else if (training_procces_finished) {
        GuiUnlock();
        dr_gui_dim(work_area);
        const char* message = TextFormat("The neural network has been successfully trained with a last error: "
            DR_APPLICATION_TEXT_FORMAT_PRECISION, training_error);
        const int message_box_result = GuiMessageBox(message_box_bounds, "Success", message, "Ok");
        if (message_box_result >= 0) {
            training_procces_finished = false;
        } else {
            GuiLock();
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////// PREDICTION

void dr_application_prediction_tab() {
    // mouse pos
    const Vector2 mouse_pos = GetMousePosition();

    // work area
    Rectangle work_area = { 0 };
    work_area.x = 0;
    work_area.y = DR_APPLICATION_TAB_BOTTOM;
    work_area.width  = window_size.x;
    work_area.height = window_size.y - DR_APPLICATION_TAB_BOTTOM;

    // container
    Rectangle container_bounds = { 0 };
    container_bounds.width  = work_area.width / 1.1;
    container_bounds.height = DR_APPLICATION_CANVAS_HEIGHT + DR_APPLICATION_CANVAS_CLEAR_BUTTON_HEIGHT * 2;
    container_bounds.x = work_area.x + work_area.width / 2 - container_bounds.width / 2;
    container_bounds.y = work_area.y + work_area.height / 2 - container_bounds.height / 2;

    // canvas
    Rectangle canvas_bounds = { 0 };
    canvas_bounds.x = container_bounds.x;
    canvas_bounds.y = container_bounds.y + container_bounds.height / 2 - DR_APPLICATION_CANVAS_HEIGHT / 2;
    canvas_bounds.width  = DR_APPLICATION_CANVAS_WIDTH;
    canvas_bounds.height = DR_APPLICATION_CANVAS_HEIGHT;

    // canvas connection
    const float canvas_connection_height = canvas_bounds.height / 3;
    const float canvas_connection_offset = canvas_connection_height / DR_APPLICATION_DIGITS_COUNT;
    Vector2 canvas_connection_point = { 0 };
    canvas_connection_point.x = canvas_bounds.x + canvas_bounds.width;
    canvas_connection_point.y = canvas_bounds.y + canvas_bounds.height / 2 - canvas_connection_height / 2;

    // check box
    Vector2 check_box_size = { 0 };
    check_box_size.x = 20;
    check_box_size.y = 20;
    const float check_box_margin = 10;

    // check box my nn
    Rectangle check_box_my_nn_bounds = { 0 };
    check_box_my_nn_bounds.x = container_bounds.x;
    check_box_my_nn_bounds.y =
        work_area.y + (canvas_bounds.y - work_area.y) / 2 - check_box_size.y - check_box_margin / 2;
    check_box_my_nn_bounds.width  = check_box_size.x;
    check_box_my_nn_bounds.height = check_box_size.y;

    // check box pretrained nn
    Rectangle check_box_pretrained_nn_bounds = { 0 };
    check_box_pretrained_nn_bounds.x = check_box_my_nn_bounds.x;
    check_box_pretrained_nn_bounds.y = check_box_my_nn_bounds.y + check_box_my_nn_bounds.height + check_box_margin;
    check_box_pretrained_nn_bounds.width  = check_box_size.x;
    check_box_pretrained_nn_bounds.height = check_box_size.y;

    // button predict
    Rectangle button_predict_bounds = { 0 };
    button_predict_bounds.x = canvas_bounds.x;
    button_predict_bounds.y = canvas_bounds.y + canvas_bounds.height + DR_APPLICATION_CANVAS_CLEAR_BUTTON_HEIGHT;
    button_predict_bounds.width  = canvas_bounds.width;
    button_predict_bounds.height = DR_APPLICATION_CANVAS_CLEAR_BUTTON_HEIGHT;

    // result predict bounds
    Rectangle result_predict_bounds = { 0 };
    result_predict_bounds.width  = container_bounds.width / 8;
    result_predict_bounds.height = container_bounds.height / 3;
    result_predict_bounds.x = container_bounds.x + container_bounds.width - result_predict_bounds.width;
    result_predict_bounds.y = container_bounds.y + container_bounds.height / 2 - result_predict_bounds.height / 2;

    // result connection
    const float result_connection_height = result_predict_bounds.height / 3;
    const float result_connection_offset = result_connection_height / DR_APPLICATION_DIGITS_COUNT;
    Vector2 result_connection = { 0 };
    result_connection.x = result_predict_bounds.x;
    result_connection.y = result_predict_bounds.y + result_predict_bounds.height / 2 - result_connection_height / 2;

    // circle column
    const float circle_radius = 20;
    const float circle_margin = 4;
    const float circle_column_height = (circle_radius * 2 + circle_margin) * (DR_APPLICATION_DIGITS_COUNT - 1);
    const float dist_from_canvas_to_result = result_predict_bounds.x - (canvas_bounds.x + canvas_bounds.width);
    Vector2 circle_column_pos = { 0 };
    circle_column_pos.x = (canvas_bounds.x + canvas_bounds.width) + dist_from_canvas_to_result / 2;
    circle_column_pos.y = container_bounds.y + container_bounds.height / 2 - circle_column_height / 2;

    // message box
    Rectangle message_box_bounds = { 0 };
    message_box_bounds.width  = work_area.width / 1.6;
    message_box_bounds.height = work_area.height / 4;
    message_box_bounds.x = work_area.x + work_area.width / 2 - message_box_bounds.width / 2;
    message_box_bounds.y = work_area.y + work_area.height / 2 - message_box_bounds.height / 2;

    // gui
    const bool gui_was_locked = GuiIsLocked();
    if (user_neural_network.layers_count == 0) {
        if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON) &&
            CheckCollisionPointRec(mouse_pos, check_box_my_nn_bounds) && !gui_was_locked) {
            prediction_attempt_to_select_my_neural_network_failed = true;
        }
        GuiLock();
    } 
    if (GuiCheckBox(check_box_my_nn_bounds, "My neural netowrk", prediction_use_my_neural_network)) {
        prediction_use_my_neural_network = true;
        prediction_use_pretrained_neural_network = false;
    }
    if (!gui_was_locked) {
        GuiUnlock();
    }

    if (pretrained_neural_network.layers_count == 0) {
        if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON) &&
            CheckCollisionPointRec(mouse_pos, check_box_pretrained_nn_bounds) && !gui_was_locked) {
            prediction_attempt_to_select_pretrained_neural_network_failed = true;
        }
        GuiLock();
    } 
    if (GuiCheckBox(
            check_box_pretrained_nn_bounds, "Pretrained neural network", prediction_use_pretrained_neural_network)) {
        prediction_use_my_neural_network = false;
        prediction_use_pretrained_neural_network = true;
    }
    if (!gui_was_locked) {
        GuiUnlock();
    }

    for (size_t i = 0; i < DR_APPLICATION_DIGITS_COUNT; ++i) {
        const DR_FLOAT_TYPE current_prob = prediction_probs[i];
        Vector2 circle_pos = { 0 };
        circle_pos.x = circle_column_pos.x;
        circle_pos.y = circle_column_pos.y + (circle_radius * 2 + circle_margin) * i;

        // connections: canvas - circles
        const float connection_thick = 2.0f + current_prob * 3.0f;
        const float color_val = 100 + current_prob * 155;
        const Color color = { color_val, color_val, color_val, 255 };
        DrawLineEx(canvas_connection_point, circle_pos, connection_thick, color);
        canvas_connection_point.y += canvas_connection_offset;

        Rectangle text_prob_bounds = { 0 };
        text_prob_bounds.width  = 30;
        text_prob_bounds.height = 15;
        text_prob_bounds.x =
            canvas_connection_point.x + (circle_pos.x - canvas_connection_point.x) / 2 - text_prob_bounds.width / 2;
        text_prob_bounds.y =
            canvas_connection_point.y + (circle_pos.y - canvas_connection_point.y) / 2 - text_prob_bounds.height / 2;

        const char* prob_text = prediction_show ? TextFormat("%.2f", prediction_probs[i]) : "?";
        GuiDummyRec(text_prob_bounds, prob_text);

        // connection: circles - result
        DrawLineEx(circle_pos, result_connection, connection_thick, color);
        result_connection.y += result_connection_offset;

        // circles
        DrawCircleV(circle_pos, circle_radius, BLACK);
        DrawCircleLines(circle_pos.x, circle_pos.y, circle_radius, color);
        const char* text_digit = TextFormat("%d", i);
        const size_t text_digit_font_size = 16;
        const Vector2 text_digit_size = MeasureTextEx(GetFontDefault(), text_digit, text_digit_font_size, 0);
        DrawText(text_digit, circle_pos.x - text_digit_size.x / 2,
            circle_pos.y - text_digit_size.y / 2, text_digit_font_size, color);
    }

    prediction_canvas_last_point = dr_gui_canvas(
        canvas_bounds, DR_APPLICATION_CANVAS_CLEAR_BUTTON_HEIGHT, prediction_canvas_rtexture,
        DR_APPLICATION_CANVAS_DRAW_COLOR, DR_APPLICATION_CANVAS_ERASE_COLOR, prediction_canvas_last_point);

    DrawRectangleRec(result_predict_bounds, BLACK);
    DrawRectangleLinesEx(result_predict_bounds, 1, GRAY);

    if ((prediction_use_my_neural_network || prediction_use_pretrained_neural_network) &&
        GuiButton(button_predict_bounds, "Predict")) {
        prediction_show = true;
        DR_FLOAT_TYPE pixels[DR_APPLICATION_CANVAS_PIXELS_COUNT] = { 0 };
        dr_application_canvas_get_pixels(prediction_canvas_rtexture, pixels);
        if (prediction_use_my_neural_network) {
            dr_neural_network_prediction_write(user_neural_network, pixels, prediction_probs);
        } else {
            dr_neural_network_prediction_write(pretrained_neural_network, pixels, prediction_probs);
        }
        prediction_max_prob = -1;
        for (size_t i = 0; i < DR_APPLICATION_DIGITS_COUNT; ++i) {
            const DR_FLOAT_TYPE curr_prob = prediction_probs[i];
            if (prediction_probs[i] > prediction_max_prob) {
                prediction_max_prob = curr_prob;
                prediction_predicted_digit = i;
            }
        }
    }

    const size_t prediction_font_size = 50;
    const char* prediction_text = prediction_show ? TextFormat("%d", prediction_predicted_digit) : "?";
    const Vector2 prediction_text_size = MeasureTextEx(GetFontDefault(), prediction_text, prediction_font_size, 0);
    Vector2 prediction_text_pos = { 0 };
    prediction_text_pos.x = result_predict_bounds.x + result_predict_bounds.width / 2 - prediction_text_size.x / 2;
    prediction_text_pos.y = result_predict_bounds.y + result_predict_bounds.height / 2 - prediction_text_size.y / 2;
    DrawText(prediction_text, prediction_text_pos.x, prediction_text_pos.y, prediction_font_size, WHITE);

    if (prediction_attempt_to_select_my_neural_network_failed) {
        GuiUnlock();
        dr_gui_dim(work_area);
        const int message_box_result = GuiMessageBox(message_box_bounds,
            "Your neural network is not available",
            "Create and train your neural network before using ", "Ok");
        if (message_box_result >= 0) {
            prediction_attempt_to_select_my_neural_network_failed = false;
        } else {
            GuiLock();
        }
    } else if (prediction_attempt_to_select_pretrained_neural_network_failed) {
        GuiUnlock();
        dr_gui_dim(work_area);
        const int message_box_result = GuiMessageBox(message_box_bounds,
            "The pretrained neural network is not available",
            "The pretrained neural network was not loaded ", "Ok");
        if (message_box_result >= 0) {
            prediction_attempt_to_select_pretrained_neural_network_failed = false;
        } else {
            GuiLock();
        }
    }
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
    srand(time(NULL));

    InitWindow(DR_APPLICATION_WINDOW_WIDTH, DR_APPLICATION_WINDOW_HEIGHT, DR_APPLICATION_DIGIT_RECOGNIZER_STR);
    SetTargetFPS(30);

    GuiSetStyle(DEFAULT, TEXT_SPACING, 2);

    // dataset
    dataset_canvas_rtexture = LoadRenderTexture(
        DR_APPLICATION_CANVAS_RESOLUTION_WIDTH, DR_APPLICATION_CANVAS_RESOLUTION_HEIGHT);
    dr_application_canvas_clear(dataset_canvas_rtexture);

    // training
    dr_application_training_add_hidden_layer(DR_APPLICATION_CANVAS_PIXELS_COUNT, DR_APPLICATION_TRAINING_SIGMOID_STR);
    dr_application_training_add_hidden_layer(DR_APPLICATION_CANVAS_PIXELS_COUNT, DR_APPLICATION_TRAINING_SIGMOID_STR); 

    // prediction
    prediction_canvas_rtexture = LoadRenderTexture(
        DR_APPLICATION_CANVAS_RESOLUTION_WIDTH, DR_APPLICATION_CANVAS_RESOLUTION_HEIGHT);
    dr_application_canvas_clear(prediction_canvas_rtexture);

    // application
    pretrained_neural_network = dr_neural_network_load_from_file(DR_APPLICATION_LOAD_PRETRAINED_NEURAL_NETWORK_PATH);
    if (!dr_neural_network_valid(pretrained_neural_network)) {
        dr_print_error("Error to load the pretrained neural network\n");
    }
}

void dr_application_close() {
    // training
    if (training_process_active) {
        training_process_active = false;
        const bool thread_join_result = dr_thread_join(training_thread_handle, training_thread_id);
        DR_ASSERT_MSG(thread_join_result, "thread join error when closing the application");
    }

    if (training_thread_handle && !dr_thread_close(training_thread_handle)) {
        dr_print_error("thread close error in the application");
    }

    dr_application_training_hidden_layers_info_clear();
    if (dr_neural_network_valid(user_neural_network)) {
        dr_neural_network_free(&user_neural_network);
    }
    if (dr_neural_network_valid(pretrained_neural_network)) {
        dr_neural_network_free(&pretrained_neural_network);
    }

    // dataset
    dr_application_dataset_clear();
    UnloadRenderTexture(dataset_canvas_rtexture);

    // prediction
    UnloadRenderTexture(prediction_canvas_rtexture);

    // raylib window
    CloseWindow();
}

bool dr_is_cpu_low_endian() {
    const int16_t number = 1;
    return ((const unsigned char*)&number)[0] == 1;
}

void dr_swap_bytes(unsigned char* left, unsigned char* right) {
    unsigned char temp = *left;
    *left  = *right;
    *right = temp;
}

uint32_t dr_reverse_bytes_32(uint32_t value) {
    unsigned char* bytes = (unsigned char*)&value;
    dr_swap_bytes(bytes, bytes + 3);
    dr_swap_bytes(bytes + 1, bytes + 2);
    return *((uint32_t*)bytes);
}

bool dr_application_mnist_to_dataset(const char* images, const char* labels, const char* dataset) {
    const bool cpu_low_endian = dr_is_cpu_low_endian();

    FILE* file_dataset = fopen(dataset, "wb");
    if (!file_dataset) {
        dr_print_error("Open file dataset error\n");
        return false;
    }

    // IMAGES
    FILE* file_images = fopen(images, "rb");
    if (!file_images) {
        dr_print_error("Open file images error\n");
        fclose(file_dataset);
        return false;
    }

    // read images header
    uint32_t file_images_header[4] = { 0 };
    fread(file_images_header, sizeof(file_images_header), 1, file_images);
    if (cpu_low_endian) {
        for (size_t i = 0; i < DR_ARRAY_LENGTH(file_images_header); ++i) {
            file_images_header[i] = dr_reverse_bytes_32(file_images_header[i]);
        }
    }
    const uint32_t number_of_images  = file_images_header[1];
    const uint32_t number_of_rows    = file_images_header[2];
    const uint32_t number_of_columns = file_images_header[3];

    // write dataset header
    fwrite(&number_of_images, sizeof(uint32_t), 1, file_dataset);
    fwrite(&number_of_rows, sizeof(uint32_t), 1, file_dataset);
    fwrite(&number_of_columns, sizeof(uint32_t), 1, file_dataset);

    // read images pixels
    const size_t number_of_pixels_total = number_of_rows * number_of_columns * number_of_images;
    const size_t pixels_buffer_size = sizeof(unsigned char) * number_of_pixels_total;
    unsigned char* pixels_buffer = (unsigned char*)DR_MALLOC(pixels_buffer_size);
    fread(pixels_buffer, pixels_buffer_size, 1, file_images);
    fclose(file_images);

    // write pixels to dataset
    fwrite(pixels_buffer, pixels_buffer_size, 1, file_dataset);
    DR_FREE(pixels_buffer);

    // LABELS
    FILE* file_labels = fopen(labels, "rb");
    if (!file_labels) {
        dr_print_error("Open file labels error\n");
        fclose(file_dataset);
        return false;
    }

    // read labels header
    uint32_t file_labels_header[2] = { 0 };
    fread(file_labels_header, sizeof(file_labels_header), 1, file_labels);
    if (cpu_low_endian) {
        for (size_t i = 0; i < 2; ++i) {
            file_labels_header[i] = dr_reverse_bytes_32(file_labels_header[i]);
        }
    }
    if (number_of_images != file_labels_header[1]) {
        dr_print_error("the number of elements in file labels does not equal to the number of images in file images");
        fclose(file_labels);
        fclose(file_dataset);
        return false;
    }

    // read labels
    const size_t labels_buffer_size = sizeof(unsigned char) * number_of_images;
    unsigned char* labels_buffer = (unsigned char*)DR_MALLOC(labels_buffer_size);
    fread(labels_buffer, labels_buffer_size, 1, file_labels);
    fclose(file_labels);

    // write labels to dataset
    fwrite(labels_buffer, labels_buffer_size, 1, file_dataset);
    DR_FREE(labels_buffer);
    fclose(file_dataset);

    return true;
}