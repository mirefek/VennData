import math
from gui_tool import ReorderLabels

class Legend:
    def __init__(self, gui):
        self.gui = gui
        self.graph = gui.graph
        self.enabled = "class_labels" in self.graph.metadata
        self.left = None
        self.right = None
        self.splits = None

    def draw(self, cr, color_bullets):
        if not self.enabled: return
        labels = self.graph.metadata.get("class_labels")
        if not labels:
            print("No class labels avaiable")
            self.enabled = False
            return
        
        n = len(labels)

        highlight_index = None
        if isinstance(self.gui.tool, ReorderLabels):
            highlight_index = self.gui.tool.dragged-1
        elif self.gui.active_class is not None and self.gui.active_class <= n:
            highlight_index = self.gui.active_class-1

        baseline = 30
        offset = 30
        box_offset = 10
        hl_offset = 7
        box_left_offset = 37

        cr.set_font_size(20)

        # bounding box coordinates from the basepoint of the first label
        left = math.inf
        right = -math.inf

        ups = []
        downs = []
        for i,label in enumerate(labels):
            x,y,width,height,_,_ = cr.text_extents(label)
            left = min(left,x)
            right = max(right, x+width)
            ups.append(y+i*baseline)
            downs.append(y+height+i*baseline)

        up = ups[0]
        down = downs[-1]

        start_x = self.gui.win_size[0]+left-right-offset
        start_y = offset - up

        self.splits = [start_y+ups[0]]
        self.splits.extend(
            start_y + (a+b)/2
            for a,b in zip(ups[1:], downs[:-1])
        )
        self.splits.append(start_y+downs[-1])
        self.left = start_x + left
        self.right = start_x + right

        # display shadow
        
        cr.rectangle(
            start_x-left-box_left_offset,
            start_y+up-box_offset,
            right-left+box_left_offset+box_offset,
            down-up+2*box_offset,
        )
        cr.set_source_rgba(1, 1, 1, 0.5)
        cr.fill()

        if highlight_index is not None:
            cr.rectangle(
                start_x-left-hl_offset,
                start_y+ups[highlight_index]-hl_offset,
                right-left + 2*hl_offset,
                downs[highlight_index]-ups[highlight_index] + 2*hl_offset,
            )
            cr.set_source_rgba(0.8, 0.8, 0.8)
            cr.fill()

        # display labels

        cr.set_source_rgb(0, 0, 0)
        for i,label in enumerate(labels):
            cr.move_to(start_x, start_y + i*baseline)
            cr.show_text(label)

        # display bullets

        for i in range(n):
            x = start_x - 20
            y = start_y + i*baseline - 7
            cr.arc(x,y, 12, 0, 2*math.pi)
            cr.set_source_rgb(0, 0, 0)
            cr.fill()
            if color_bullets:
                cr.set_source_rgb(*self.gui.get_color(i+1))
                cr.arc(x,y, 10, 0, 2*math.pi)
                cr.fill()                
            else:
                cr.set_source_rgb(1, 1, 1)
                self.show_text_center(cr,x,y,str(i+1))

    def pixel_to_class(self, pixel, require_in = True):
        if not self.enabled: return None
        if self.splits is None: return None
        x,y = pixel
        if require_in:
            if y < self.splits[0]: return None
            if y > self.splits[-1]: return None
            if x < self.left: return None
            if x > self.right: return None
        for i,split in enumerate(self.splits[1:-1]):
            if y < split: return i+1
        return len(self.splits)-1

    @staticmethod
    def show_text_center(cr,cx,cy,text):
        x,y,width,height,_,_ = cr.text_extents(text)
        cr.move_to(cx-x-width/2, cy-y-height/2)
        cr.show_text(text)
