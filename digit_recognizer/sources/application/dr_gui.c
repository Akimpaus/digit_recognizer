#include <application/dr_gui.h>
#define RAYGUI_IMPLEMENTATION
#include <raygui.h>

void dr_gui_dim(const Rectangle bounds) {
    DrawRectangleRec(bounds, CLITERAL(Color){ 0, 0, 0, 100 });
}

Vector2 dr_gui_canvas(const Rectangle bounds, const float button_clear_height, RenderTexture2D target,
    const Color draw_color, const Color erase_color, Vector2 last_point) {
    SetMouseOffset(-bounds.x, -bounds.y);
    SetMouseScale(target.texture.width / bounds.width, target.texture.height / bounds.height);
    const Vector2 mouse_pos_canvas = GetMousePosition();

    const float bounds_outline_margin = 1;
    const float line_thick = 1;
    const Rectangle bounds_outline_rect = {
        bounds.x - bounds_outline_margin,
        bounds.y - bounds_outline_margin,
        bounds.width + bounds_outline_margin * 2,
        bounds.height + bounds_outline_margin * 2
    };
    DrawRectangleLinesEx(bounds_outline_rect, line_thick, GRAY);

    BeginTextureMode(target);

    const bool mouse_left_down  = IsMouseButtonDown(MOUSE_LEFT_BUTTON);
    const bool mouse_right_down = IsMouseButtonDown(MOUSE_RIGHT_BUTTON);
    if (mouse_left_down || mouse_right_down) {
        if (last_point.x == -1 || last_point.y == -1) {
            last_point = mouse_pos_canvas;
        }
        const Color line_color = mouse_left_down ? draw_color : erase_color;
        DrawLineV(mouse_pos_canvas, last_point, line_color);
        last_point = mouse_pos_canvas;
    } else if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT) || IsMouseButtonReleased(MOUSE_BUTTON_RIGHT)) {
        last_point.x = -1;
        last_point.y = -1;
    }

    EndTextureMode();

    const Rectangle rect_src = { 0, 0, target.texture.width, -target.texture.height };
    DrawTexturePro(target.texture, rect_src, bounds, CLITERAL(Vector2){ 0, 0 }, 0, WHITE );

    SetMouseOffset(0, 0);
    SetMouseScale(1, 1);

    const Rectangle bounds_button_clear = {
        bounds.x,
        bounds.y + bounds.height + line_thick,
        bounds.width,
        button_clear_height
    };

    if (GuiButton(bounds_button_clear, "Clear")) {
        BeginTextureMode(target);
        ClearBackground(erase_color);
        EndTextureMode();
    }

    return last_point;
}

int dr_gui_numeric_buttons_row(const Rectangle bounds, const size_t count, const size_t* values) {
    int clicked = -1;

    if (count == 0) {
        return clicked;
    }

    const Vector2 button_size = { 35, 35 };
    const float button_space  = 10;
    const float total_button_space = button_space * (count - 1);
    const float total_button_width = button_size.x * count;
    const float left_pos = (bounds.x + bounds.width) / 2 - total_button_width / 2;

    for (size_t i = 0; i < count; ++i) {
        const Rectangle button_bounds = {
            left_pos + i * (button_size.x + button_space) - total_button_space / 2,
            bounds.y + bounds.height / 2 - button_size.y / 2,
            button_size.x,
            button_size.y
        };
        if (GuiButton(button_bounds, TextFormat("%d", i))) {
            clicked = i;
        }

        const char* label_text = TextFormat("%d", values[i]);
        const Font label_text_font       = GetFontDefault();
        const float label_text_font_size = 10;
        const float label_text_spacing   = 0;
        const Vector2 measure_label_text = MeasureTextEx(
            label_text_font, label_text, label_text_font_size, label_text_spacing);
        const Vector2 label_text_position = {
            button_bounds.x + button_bounds.width / 2 - measure_label_text.x / 2,
            button_bounds.y + button_bounds.height + measure_label_text.y / 2
        };
        DrawTextEx(
            label_text_font, label_text, label_text_position, label_text_font_size, label_text_spacing, LIGHTGRAY);
    }

    return clicked;
}