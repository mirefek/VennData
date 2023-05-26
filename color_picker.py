import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk
import numpy as np

class ColorPicker(Gtk.Window):
    def __init__(self, main):
        super().__init__()
        self.main = main
        if main is None:
            self.colors = [
                np.array([1,  0,  0.6]),
                np.array([0,  0.2,1]),
                np.array([0,  1,  0]),
                np.array([1, 0.6, 0]),
                np.array([0.8, 0.8, 0]),
            ]
        else:
            self.colors = main.colors
        self.index = None
        self.colorchooser = Gtk.ColorSelection()
        self.vbox = Gtk.VBox()
        self.hbox = Gtk.HBox()
        self.vbox.add(self.colorchooser)
        self.vbox.add(self.hbox)
        self.frames = [
            Gtk.DrawingArea()
            for _ in self.colors
        ]
        self.buttons = [
            Gtk.ToggleButton()
            for _ in self.colors
        ]
        for i,(color,frame,button) in enumerate(zip(self.colors, self.frames, self.buttons)):
            # frame.override_background_color(Gtk.StateFlags.NORMAL, Gdk.RGBA(*color, 1.0))
            button.add(frame)
            self.hbox.add(button)
            frame.connect("draw", self.on_button_draw, i)
            button.connect("toggled", self.on_button_toggled, i)
            button.drag_source_set(Gdk.ModifierType.BUTTON1_MASK, [], Gdk.DragAction.COPY)
            button.drag_source_add_text_targets()
            button.drag_dest_set(Gtk.DestDefaults.ALL, [], Gdk.DragAction.COPY)
            button.drag_dest_add_text_targets()
            button.connect("drag-data-get", self.on_drag_data_get, i)
            button.connect("drag-data-received", self.on_drag_data_received, i)

        self.buttons[0].set_active(True)
        ok_button = Gtk.Button(label = "OK")
        ok_button.connect("clicked", self.on_ok_clicked)
        self.vbox.add(ok_button)
        self.add(self.vbox)
        self.show_all()
        self.colorchooser.connect("color-changed", self.on_change_color)
        self.connect("key-press-event", self.on_key_press)

    def on_button_toggled(self, button, index):
        if not button.get_active(): return
        if self.index is not None:
            self.buttons[self.index].set_active(False)
        self.index = index
        self.colorchooser.set_current_rgba(Gdk.RGBA(*self.colors[self.index], 1.0))
        self.colorchooser.set_previous_rgba(Gdk.RGBA(*self.colors[self.index], 1.0))

    def on_change_color(self,w):
        if self.index is None: return
        color = self.colorchooser.get_current_rgba()
        self.colors[self.index] = np.array(list(color)[:3])
        self.frames[self.index].queue_draw()
        if self.main is not None: self.main.darea.queue_draw()

    def on_drag_data_get(self,w,ctx,data,info,time,index):
        data.set_text(str(index), -1)
    def on_drag_data_received(self,w,ctx,x,y,data,info,time,index):
        index2 = data.get_text()
        if not index2.isnumeric(): return
        index2 = int(index2)
        if not 0 <= index2 < len(self.colors): return
        if index == index2: return
        self.colors[index], self.colors[index2] = self.colors[index2], self.colors[index]
        self.frames[index].queue_draw()
        self.frames[index2].queue_draw()
        if self.main is not None: self.main.darea.queue_draw()
        
        if self.index in (index, index2):
            self.colorchooser.set_current_rgba(Gdk.RGBA(*self.colors[self.index], 1.0))

    def on_key_press(self,w,e):
        keyval_name = Gdk.keyval_name(e.keyval)
        if keyval_name == "Escape":
            if self.main is None: Gtk.main_quit()
            else: self.close()

    def on_ok_clicked(self,w):
        if self.main is None: Gtk.main_quit()
        else: self.close()        

    def on_button_draw(self,widget, cr, index):
        width = widget.get_allocated_width()
        height = widget.get_allocated_height()
        color = self.colors[index]

        cr.rectangle(0, 0, width, height)
        cr.set_source_rgba(0,0,0)
        cr.fill()
        cr.rectangle(1, 1, width-2, height-2)
        cr.set_source_rgba(*color)
        cr.fill()

    def close(self):
        if self.main is None: Gtk.main_quit()
        else:
            self.main.color_picker = None
            self.destroy()

if __name__ == "__main__":
    ColorPicker(None)
    Gtk.main()
