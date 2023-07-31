#include <application/dr_gui.h>
#define RAYGUI_IMPLEMENTATION
#include <raygui.h>
#include <ctype.h>

void dr_gui_dim(const Rectangle bounds) {
    DrawRectangleRec(bounds, CLITERAL(Color){ 0, 0, 0, 100 });
}

Vector2 dr_gui_canvas(const Rectangle bounds, const float button_clear_height, RenderTexture2D target,
    const Color draw_color, const Color erase_color, Vector2 last_point) {
    SetMouseOffset(-bounds.x, -bounds.y);
    SetMouseScale(target.texture.width / bounds.width, target.texture.height / bounds.height);

    const Vector2 mouse_pos = GetMousePosition();

    // canvas
    const float bounds_outline_margin = 1;
    const float line_thick = 1;
    Rectangle bounds_outline_rect = { 0 };
    bounds_outline_rect.x = bounds.x - bounds_outline_margin;
    bounds_outline_rect.y = bounds.y - bounds_outline_margin;
    bounds_outline_rect.width  = bounds.width + bounds_outline_margin * 2;
    bounds_outline_rect.height = bounds.height + bounds_outline_margin * 2;

    // button clear
    Rectangle bounds_button_clear = { 0 };
    bounds_button_clear.x = bounds.x;
    bounds_button_clear.y = bounds.y + bounds.height + line_thick;
    bounds_button_clear.width  = bounds.width;
    bounds_button_clear.height = button_clear_height;

    // target
    Rectangle target_rect_src = { 0 };
    target_rect_src.x = 0;
    target_rect_src.y = 0;
    target_rect_src.width  = target.texture.width;
    target_rect_src.height = -target.texture.height;

    // gui
    DrawRectangleLinesEx(bounds_outline_rect, line_thick, GRAY);

    if (!GuiIsLocked()) {
        BeginTextureMode(target);
        const bool mouse_left_down  = IsMouseButtonDown(MOUSE_LEFT_BUTTON);
        const bool mouse_right_down = IsMouseButtonDown(MOUSE_RIGHT_BUTTON);
        if (mouse_left_down || mouse_right_down) {
            if (last_point.x == -1 || last_point.y == -1) {
                last_point = mouse_pos;
            }
            const Color line_color = mouse_left_down ? draw_color : erase_color;
            DrawLineEx(mouse_pos, last_point, 1.5, line_color);
            last_point = mouse_pos;
        } else if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT) || IsMouseButtonReleased(MOUSE_BUTTON_RIGHT)) {
            last_point.x = -1;
            last_point.y = -1;
        }
        EndTextureMode();
    }

    DrawTexturePro(target.texture, target_rect_src, bounds, CLITERAL(Vector2){ 0, 0 }, 0, WHITE );

    SetMouseOffset(0, 0);
    SetMouseScale(1, 1);

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

    // button
    const Vector2 button_size = { 35, 35 };
    const float button_space  = 10;
    const float total_button_space = button_space * (count - 1);
    const float total_button_width = button_size.x * count;
    const float left_pos = (bounds.x + bounds.width) / 2 - total_button_width / 2;

    // label
    const Font label_text_font       = GetFontDefault();
    const float label_text_font_size = 10;
    const float label_text_spacing   = GuiGetStyle(DEFAULT, TEXT_SPACING);

    // gui
    for (size_t i = 0; i < count; ++i) {
        Rectangle button_bounds = { 0 };
        button_bounds.x = left_pos + i * (button_size.x + button_space) - total_button_space / 2;
        button_bounds.y = bounds.y + bounds.height / 2 - button_size.y / 2;
        button_bounds.width  = button_size.x;
        button_bounds.height = button_size.y;

        if (GuiButton(button_bounds, TextFormat("%d", i))) {
            clicked = i;
        }

        const char* label_text = TextFormat("%d", values[i]);
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