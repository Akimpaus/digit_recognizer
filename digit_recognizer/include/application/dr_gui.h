#ifndef DR_GUI_H
#define DR_GUI_H

#include <raygui.h>
#include <general/dr_utils.h>

#define RAYGUI_WINDOWBOX_STATUSBAR_HEIGHT 20

void dr_gui_dim(const Rectangle bounds);

Vector2 dr_gui_canvas(const Rectangle bounds, const float button_clear_height, RenderTexture2D target,
    const Color draw_color, const Color erase_color, Vector2 last_point);

int dr_gui_numeric_buttons_row(const Rectangle bounds, const size_t count, const size_t* values);

#endif // DR_GUI_H