#!/usr/bin/python3

import math
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GLib
import cairo
import json
import random
import itertools

from planar_graph import *
import numpy as np
from graph_cursor import GraphCursor
from gui_tool import *
from color_picker import ColorPicker
from legend import Legend

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

class VennDataGui(Gtk.Window):

    def __init__(self, graph, fname, optimizer_step, optimizer_coef, win_size = (800, 600)):
        super().__init__()
        self.color_picker = None
        self.mb_grasp = None
        self.graph = graph
        self._import_graph_metadata()
        self.ev_sensitivity = 1.0
        self.display_graph = self.display_graph_default
        self.cursor  = GraphCursor(self)
        self.fname = fname
        self.ext = ".vgr"
        self.active_class = None
        self.total_area_style = None
        self.legend = Legend(self)
        self.optimizer_step = optimizer_step
        self.optimizer_coef = optimizer_coef

        self.basic_tool = BasicTool(self)
        self.tool = self.basic_tool
        self.optimizer = None
        self.optim_timer = None

        self.darea = Gtk.DrawingArea()
        self.darea.connect("draw", self.on_draw)
        self.darea.set_events(Gdk.EventMask.BUTTON_PRESS_MASK |
                              Gdk.EventMask.BUTTON_RELEASE_MASK |
                              Gdk.EventMask.KEY_PRESS_MASK |
                              Gdk.EventMask.KEY_RELEASE_MASK |
                              Gdk.EventMask.SCROLL_MASK |
                              Gdk.EventMask.POINTER_MOTION_MASK)
        self.add(self.darea)

        self.darea.connect("button-press-event", self.on_button_press)
        self.darea.connect("button-release-event", self.on_button_release)
        self.darea.connect("scroll-event", self.on_scroll)
        self.darea.connect("motion-notify-event", self.on_motion)
        self.connect("key-press-event", self.on_key_press)
        self.connect("key-release-event", self.on_key_release)

        self.set_title(f"Venn-Data GUI -- {fname}")
        self.resize(*win_size)
        self.win_size = win_size
        self.set_position(Gtk.WindowPosition.CENTER)
        self.connect("delete-event", Gtk.main_quit)
        self.show_all()

    def update_win_size(self):
        self.win_size = (self.darea.get_allocated_width(), self.darea.get_allocated_height())

    def pixel_to_coor(self, pixel):
        px,py = pixel
        w,h = self.win_size
        sx,sy = self.shift
        x = (px - w/2) / self.scale - sx
        y = (h/2 - py) / self.scale - sy
        return (x,y)
    def coor_to_pixel(self, pos):
        w,h = self.win_size
        sx,sy = self.shift
        x,y = pos
        x = float(x)
        y = float(y)
        px = (x + sx) * self.scale + w/2
        py = h/2 - (y + sy) * self.scale
        return px,py
    def set_shift(self, pixel, coor):
        w,h = self.win_size
        px,py = pixel
        x,y = coor
        sx = (px - w/2) / self.scale - x
        sy = (h/2 - py) / self.scale - y
        self.shift = sx,sy

    def on_scroll(self,w,e):
        coor = self.pixel_to_coor((e.x, e.y))
        if e.direction == Gdk.ScrollDirection.DOWN: self.scale *= 0.9
        elif e.direction == Gdk.ScrollDirection.UP: self.scale /= 0.9
        # print("zoom {}".format(self.scale))
        self.set_shift((e.x, e.y), coor)
        self.darea.queue_draw()

    def on_motion(self,w,e):
        self.tool.on_motion((e.x,e.y))
        if e.state & Gdk.ModifierType.BUTTON2_MASK:
            if self.mb_grasp is None: return
            self.set_shift((e.x, e.y), self.mb_grasp)
            self.darea.queue_draw()

    def on_key_press(self,w,e):

        keyval = e.keyval
        keyval_name = Gdk.keyval_name(keyval)
        # do not distinguish standard and numlock key variants
        keyval_name = remove_prefix(keyval_name, "KP_")
        # print("Press:", keyval_name)
        if keyval_name == "Escape":
            Gtk.main_quit()
        elif keyval_name == 'minus':
            self.ev_sensitivity /= 2
            print(self.ev_sensitivity)
            self.darea.queue_draw()
        elif keyval_name == 'equal':
            self.ev_sensitivity *= 2
            print(self.ev_sensitivity)
            self.darea.queue_draw()
        elif keyval_name == 'space':
            self.optim_timer_stop()
            self.cursor.disable()
            self.tool = self.basic_tool
            self.active_class = None
            self.darea.queue_draw()
        elif self.cursor.move_by_key(keyval_name):
            self.darea.queue_draw()
        elif keyval_name == 'F2': self.save()
        elif keyval_name == 'F3': self.load()
        elif keyval_name.lower() in 's':
            self.optim_timer_start(keyval_name.islower(), 200)
        elif keyval_name.lower() == 'o':
            if self.optim_timer is None:
                self.optim_timer_start(keyval_name.islower(), 30)
            else:
                self.optim_timer_stop()
        elif keyval_name == 'a':
            self.analyze_graph()
        elif keyval_name == 'v':
            self.display_graph = self.display_venn_diagram
            self.darea.queue_draw()
        elif keyval_name == 'g':
            self.display_graph = self.display_graph_default
            self.darea.queue_draw()
        elif keyval_name.isnumeric() and int(keyval_name) > 0:
            self.active_class = int(keyval_name)
            self.darea.queue_draw()
        elif keyval_name == 'x':
            self.export("png")
        elif keyval_name == 'X':
            self.export("svg")
        elif keyval_name == 'c':
            if self.color_picker is not None:
                self.color_picker.close()
            if self.colors: self.color_picker = ColorPicker(self)
            else: print("There are no allocated colors yet")
        elif keyval_name == 't':
            if self.total_area_style == None: self.total_area_style = "rectangle"
            elif self.total_area_style == "rectangle": self.total_area_style = "circle"
            else: self.total_area_style = None
            self.darea.queue_draw()
        elif keyval_name == 'l':
            self.legend.enabled = not self.legend.enabled
            self.darea.queue_draw()

    def on_key_release(self,w,e):

        keyval_name = Gdk.keyval_name(e.keyval)
        if keyval_name in ('s', 'S'): self.optim_timer_stop()

    def pixel_node_distance(self, node, pixel):
        return point_distance(pixel, self.coor_to_pixel(node.pos))

    def pixel_edge_distance(self, edge, pixel):
        A,B = [
            self.coor_to_pixel(coor)
            for coor in edge.endpoints
        ]
        P, subarcs = project_to_arc(
            A,B,edge.curvature, pixel
        )
        if P is None: return math.inf
        else: return point_distance(pixel, P)

    def get_pixel_node(self, pixel):
        dist, _, closest_node = min(
            (self.pixel_node_distance(node, pixel), id(node), node)
            for node in self.graph.nodes
        )
        if dist < 20: return closest_node
        else: return None
    def get_pixel_edge(self, pixel):
        dist, _, closest_edge = min(
            (self.pixel_edge_distance(edge, pixel), id(edge), edge)
            for edge in self.graph.edges
        )
        if dist < 20: return closest_edge
        else: return None
    def get_pixel_face(self, pixel):
        coor = self.pixel_to_coor(pixel)
        for face in self.graph.faces:
            if face.contains_pos(coor):
                return face
        return None

    def bisect_edge_on_pixel(self, edge, pixel):
        self.disconnect_optimizer(keep_timer = True)
        A, B = (self.coor_to_pixel(node.pos) for node in edge.nodes)
        arc = edge.curvature
        P, subarcs = project_to_arc(A,B,arc, pixel)
        if P is None: return None
        node = edge.bisect()
        node.pos = self.pixel_to_coor(P)
        for edge, arc in zip(node.edges, subarcs):
            edge.curvature = arc
        self.darea.queue_draw()
        return node

    def on_button_press(self, w, e):
        if e.type != Gdk.EventType.BUTTON_PRESS: return
        if e.button == 1: self.tool.on_left_click((e.x, e.y))
        elif e.button == 2: self.mb_grasp = self.pixel_to_coor((e.x, e.y))
        elif e.button == 3: self.tool.on_right_click((e.x, e.y))

    def on_button_release(self, w, e):
        if e.button == 2: self.mb_grasp = None
        else: self.tool.on_release((e.x, e.y))

    def draw_edge(self, cr, edge):
        p0,p1 = [
            self.coor_to_pixel(coor)
            for coor in edge.endpoints
        ]
        cr.move_to(*p0)
        arc = edge.curvature
        draw_arc(cr, p0,p1, arc)
        cr.stroke()
    def draw_node(self, cr, node, size = 7):
        x,y = self.coor_to_pixel(node.pos)
        if node.degree == 2: size *= 0.5
        cr.arc(x, y, size, 0, 2*math.pi)
        cr.fill()
    def draw_face(self, cr, face, allow_outside = True):
        points = [
            self.coor_to_pixel(node.pos)
            for node in face.nodes
        ]
        edges = face.edges

        if allow_outside and face.turning_number_cw() < 0: # fill outer face
            xs,ys = zip(*points)
            w,h = self.win_size
            bx0 = min(min(xs), 0)
            bx1 = max(max(xs), w)
            by0 = min(min(ys), 0)
            by1 = max(max(ys), w)
            i = min(range(len(points)), key = lambda i: points[i][0])
            surrounding = [
                (bx0,by0), (bx1,by0),
                (bx1,by1), (bx0,by1),
                (bx0,by0),
            ]
            points = points[i+1:]+points[:i+1]
            edges = edges[i+1:]+edges[:i+1]

            cr.move_to(*points[-1])
            for point in surrounding: cr.line_to(*point)
            cr.line_to(*points[-1])
        else:
            cr.move_to(*points[-1])

        last_point = points[-1]
        for edge,point in zip(edges, points):
            arc = edge.curvature
            if edge.faces.index(face) == 0: arc = -arc
            draw_arc(cr, last_point, point, arc)
            last_point = point
        cr.close_path()
        cr.fill()

    def draw_total_area(self, cr):
        if self.total_area_style is None: return
        total_area = self.graph.metadata.get("total_area", None)
        if total_area is None:
            self.total_area_style = None
            print("Unknown total area")
        (x,y),(w,h) = self.graph.bounding_box()
        cx,cy = x+w/2,y-h/2
        if self.total_area_style == "rectangle":
            coef = math.sqrt(total_area / abs(w*h))
            w *= coef
            h *= coef
            x,y = cx-w/2, cy+h/2
            pw,ph = w*self.scale, h*self.scale
            cr.rectangle(*self.coor_to_pixel((x,y)),pw,ph)
            #print(w*h, pw*ph)
        elif self.total_area_style == "circle":
            radius = math.sqrt(total_area / math.pi)
            pixel_radius = self.scale * radius
            center = (0,0)
            pixel_center = self.coor_to_pixel((cx,cy))
            cr.arc(*pixel_center, pixel_radius, 0, 2*math.pi)
        else: raise Exception(f"Unexpected total area style: {self.total_area_style}")
        cr.set_source_rgb(0,0,0)
        cr.set_line_width(3)
        cr.stroke()

    def fill_background(self,cr):
        cr.rectangle(0,0,*self.win_size)
        cr.set_source_rgb(1,1,1)
        cr.fill()

    def on_draw(self, wid, cr):
        self.update_win_size()
        self.fill_background(cr)

        self.tool.display_bg(cr)
        self.cursor.display_bg(cr)

        self.display_graph(cr)

        self.cursor.display_fg(cr)
        self.tool.display_fg(cr)

    def export(self, ext):
        if self.fname.endswith(self.ext):
            fname = self.fname[:len(self.fname)-len(self.ext)]
        else: fname = self.fname
        fname = fname+'.'+ext
        if ext == "svg": self.svg_export(fname)
        elif ext == "png": self.png_export(fname)
        else: raise Exception(f"Unsupported export file extension: {ext}")
    def svg_export(self, fname):
        print(f"Saving SVG to {fname}")
        surface = cairo.SVGSurface(fname, *self.win_size)
        cr = cairo.Context(surface)
        self.display_venn_diagram(cr)
        surface.finish()
    def png_export(self, fname):
        print(f"Saving PNG to {fname}")
        surface = cairo.ImageSurface(cairo.Format.RGB24, *self.win_size)
        cr = cairo.Context(surface)
        self.fill_background(cr)
        self.display_venn_diagram(cr)
        surface.write_to_png(fname)

    def display_active_class(self, cr):
        for face in self.graph.faces:
            cr.set_source_rgba(0, 0, 0, 0.2)
            if self.active_class in face.classes:
                self.draw_face(cr, face)

    def display_graph_default(self, cr):
        if self.active_class is None:
            classes_areas = {
                classes : self.graph.get_classes_area(classes)
                for classes in self.graph.classes_to_goal_areas.keys()
            }
            for face in self.graph.faces:
                goal_area = self.graph.classes_to_goal_areas.get(face.classes, None)
                current_area = classes_areas.get(face.classes, None)
                if goal_area is None or current_area is None: continue
                elif goal_area == 0: cr.set_source_rgb(0,0.5,1)
                else:
                    diff = current_area / goal_area - 1
                    if diff > 0:
                        x = 1 - 1 / (diff*self.ev_sensitivity+1)
                        cr.set_source_rgba(0,1,0, x)
                    if diff < 0:
                        x = 1 - 1 / (-diff*self.ev_sensitivity+1)
                        cr.set_source_rgba(1,0,0, x)
                self.draw_face(cr, face, allow_outside = False)
        else:
            self.display_active_class(cr)

        cr.set_source_rgb(0,0,0)
        cr.set_line_width(3)
        for edge in self.graph.edges: self.draw_edge(cr, edge)
        cr.set_source_rgb(0,0,0)
        for node in self.graph.nodes: self.draw_node(cr, node)

        self.draw_total_area(cr)
        self.legend.draw(cr,False)

    def display_venn_diagram(self, cr):
        if self.active_class is None:
            for face in self.graph.faces:
                classes = face.classes
                if not face.classes: continue
                c = np.ones(3)
                for cl in classes:
                    c -= (1-self.get_color(cl))*0.2
                cr.set_source_rgb(*c)
                self.draw_face(cr, face)
        else:
            self.display_active_class(cr)

        cr.set_line_width(3)
        for edge in self.graph.edges:
            f0,f1 = edge.faces
            classes = f0.classes ^ f1.classes
            if len(classes) == 0:
                # print("Warning: an edge with no classes")
                cr.set_source_rgb(0,0,0)
            else:
                if len(classes) == 1:
                    [cl] = classes
                    color = self.get_color(cl)
                else:
                    n = len(classes)
                    color = np.average([self.get_color(cl) for cl in classes], axis = 0)
                cr.set_source_rgb(*color)
            self.draw_edge(cr, edge)

        self.draw_total_area(cr)
        self.legend.draw(cr,True)

    def _import_graph_metadata(self):
        class_to_color = self.graph.metadata.get("class_to_color")
        self.scale = self.graph.metadata.get("view_scale", 50)
        self.shift = tuple(self.graph.metadata.get("view_shift", (0,0)))
        if class_to_color is None: self.colors = []
        else:
            class_to_color = {
                int(cl) : np.array(color)
                for cl,color in class_to_color.items()
            }
            assert set(class_to_color.keys()) == set(range(1, len(class_to_color)+1))
            self.colors = [
                class_to_color[cl] for cl in range(1, len(class_to_color)+1)
            ]

    def load(self, fname = None):
        if fname is None: fname = self.fname
        self.disconnect_optimizer() # torch numbers cannot be stored to json
        if self.color_picker is not None: self.color_picker.close()
        self.graph.load_from_file(fname)
        self._import_graph_metadata
        self.darea.queue_draw()

    def save(self, fname = None):
        if fname is None: fname = self.fname
        self.disconnect_optimizer() # torch numbers cannot be stored to json
        self.graph.metadata["view_scale"] = self.scale
        self.graph.metadata["view_shift"] = list(self.shift)
        self.graph.metadata["class_to_color"] = {
            str(i+1) : list(color)
            for i,color in enumerate(self.colors)
        }
        print(f"Saving to {fname}")
        self.graph.dump_to_file(fname)

    def get_color(self, class_i):
        assert isinstance(class_i, int) and class_i > 0
        while class_i > len(self.colors):
            color = [0,1,random.random()]
            random.shuffle(color)
            self.colors.append(np.array(color))
        return self.colors[class_i-1]

    def disconnect_optimizer(self, keep_timer = False):
        if self.optimizer is not None:
            if not keep_timer: self.optim_timer_stop()
            self.optimizer.disconnect()
    def optim_timer_stop(self):
        if self.optim_timer is None: return
        GLib.source_remove(self.optim_timer)
        self.optim_timer = None
    def optim_timer_start(self, area_only, first_time):
        if self.optim_timer is not None: return
        if self.optimizer is None:
            print("Loading optimizer")
            from torch_graph import PlanarGraphOptimizer
            self.optimizer = PlanarGraphOptimizer()
        self.optim_timer_stop()
        self.optimizer_make_step(area_only)
        self.optim_timer = GLib.timeout_add(first_time, self.optim_timer_step, area_only)
    def optim_timer_step(self, area_only):
        self.optimizer_make_step(area_only)
        GLib.source_remove(self.optim_timer)
        self.optim_timer = GLib.timeout_add(30, self.optim_timer_step, area_only)
        return False
    def optimizer_make_step(self, area_only):
        if not self.optimizer.connected: self.optimizer.connect(self.graph)
        if area_only: self.optimizer.gradient_step(step = self.optimizer_step)
        else: self.optimizer.gradient_step_min_len(
                step = self.optimizer_step,
                coef = self.optimizer_coef,
        )
        self.darea.queue_draw()

    def analyze_graph(self):
        missing = []
        extra = []
        classes_s = set(
            classes
            for classes,faces in self.graph.classes_to_faces.items()
            if faces
        )
        if self.graph.classes_to_goal_areas:
            classes_s.update(self.graph.classes_to_goal_areas.keys())
            for classes in classes_s:
                is_goal = (
                    classes in self.graph.classes_to_goal_areas and
                    self.graph.classes_to_goal_areas[classes] != 0
                )
                is_current = (
                    classes in self.graph.classes_to_faces and
                    self.graph.classes_to_faces[classes]
                )
                if is_goal and not is_current:
                    missing.append(tuple(sorted(classes)))
                elif is_current and not is_goal:
                    extra.append(tuple(sorted(classes)))
            if missing:
                print("Missing:")
                missing.sort(key = lambda x: (len(x),x))
                for x in missing:
                    print(' ', x)
            if extra:
                print("Extra:")
                extra.sort(key = lambda x: (len(x),x))
                for x in extra:
                    print(' ', x)
            if not (missing or extra):
                print("Existing class combinations correspond to the data")
        else:
            all_classes = self.graph.all_classes()
            missing = []
            for classes in itertools.product(*([[], [x]] for x in all_classes)):
                classes = list(itertools.chain.from_iterable(classes))
                if not classes: continue
                if frozenset(classes) not in classes_s: missing.append(classes)
            if missing:
                print("Missing:")
                missing.sort(key = lambda x: (len(x),x))
                for x in missing: print(' ', x)
            else:
                print(f"All combinations of {all_classes} occur")
        for node in self.graph.nodes:
            edge = node.find_wrong_edge_order()
            if edge is not None:
                self.cursor.node_or_face = node
                self.cursor.edge = edge
                print("Node with incorrect edge order")
                self.darea.queue_draw()
                break
        else:
            print("All nodes have correctly ordered edges")

if __name__ == "__main__":

    import argparse
    import os

    data_fnames = [
        "tactician.bench",
        # "graph2tac-all-names.bench",
        # "sauto.bench",
        "coqhammer-cvc4.bench",
        "transformer.bench",
        "graph2tac-new-names.bench",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", nargs='?', default = "workfile.vgr", type=str,
                        help="file to load / save, default: workfile.vgr")
    parser.add_argument("-d", "--data", default = None, type = str,
                        help="filename for json statistics created with venn.py")
    parser.add_argument("-s", "--step", default = None, type = float,
                        help="step size for the optimizer, default: 0.01")
    parser.add_argument("-c", "--coef", default = None, type = float,
                        help="coefficient between length and area loss for the optimizer, default: 5.0")
    args = parser.parse_args()

    graph = PlanarGraph()
    graph.init_triangle()
    if os.path.isfile(args.filename):
        graph.load_from_file(args.filename)

    if args.data is not None:
        with open(args.data) as f:
            data = json.load(f)
        graph.set_goal_areas(data)

    win = VennDataGui(
        graph = graph,
        fname = args.filename,
        optimizer_step = args.step,
        optimizer_coef = args.coef,
    )
    Gtk.main()
