import math
from arc_geometry import *

class Tool:
    def on_motion(self, pixel):
        pass
    def on_left_click(self, pixel):
        self.gui.tool = self.gui.basic_tool
        self.gui.darea.queue_draw()
    def on_right_click(self, pixel):
        self.gui.tool = self.gui.basic_tool
        self.gui.darea.queue_draw()
    def on_release(self, pixel):
        pass
    def display_bg(self, cr):
        pass
    def display_fg(self, cr):
        pass

class BasicTool(Tool):
    def __init__(self, gui):
        self.gui = gui

    def on_left_click(self, pixel):
        gui = self.gui
        node = gui.get_pixel_node(pixel)
        click_f = None
        tool_on_move = None
        cl = self.gui.legend.pixel_to_class(pixel)
        if cl is not None:
            gui.tool = ReorderLabels(self.gui, cl)
            gui.darea.queue_draw()
            return
        if gui.active_class is not None or gui.cursor.face is not None:
            face = self.gui.get_pixel_face(pixel)
            if face is not None:
                if gui.cursor.face is not None:
                    def click_f():
                        gui.cursor.node_or_face = face
                        gui.cursor.edge = face.edges[0]
                        gui.darea.queue_draw()
                else:
                    def click_f():
                        if gui.active_class in face.classes:
                            face.remove_class(gui.active_class)
                        else:
                            face.add_class(gui.active_class)
                        gui.darea.queue_draw()

        if node is not None:
            if click_f is None:
                def click_f():
                    self.gui.tool = ConstructSplit(self.gui, node, pixel)
            tool_on_move = lambda pixel: MoveTool(self.gui, node, pixel)
        else:
            edge = gui.get_pixel_edge(pixel)
            if edge is not None:
                if click_f is None:
                    def click_f():
                        self.gui.bisect_edge_on_pixel(edge, pixel)
                tool_on_move = lambda pixel: MoveEdgeTool(self.gui, edge, pixel)

        if tool_on_move is not None:
            gui.tool = GeneralMoveOrRelease(
                gui, pixel,
                tool_on_move = tool_on_move,
                f_on_release = click_f,
            )
        elif click_f != None:
            click_f()

    def on_right_click(self, pixel):
        gui = self.gui
        node = gui.get_pixel_node(pixel)
        if node is not None:
            X = node.pos
            edge = node.unbisect()
            if edge is not None:
                gui.disconnect_optimizer(keep_timer = True)
                A,B = edge.endpoints
                edge.curvature = arc_from_point(A,B,X)
                gui.darea.queue_draw()
            else:
                print("cannot remove this node")
        else:
            edge = gui.get_pixel_edge(pixel)
            if edge is not None:
                f0,f1 = edge.faces
                if abs(f0.area()) >= abs(f1.area()):
                    classes = f0.classes
                else:
                    classes = f1.classes
                face = edge.join()
                if face is not None:
                    gui.disconnect_optimizer(keep_timer = True)
                    face.set_classes(classes)
                    gui.darea.queue_draw()
                else:
                    print("cannot break this edge")

class MoveTool(Tool):
    def __init__(self, gui, node, pixel = None):
        self.gui = gui
        self.node = node
        if pixel is not None: self.on_motion(pixel)
    def on_motion(self, pixel):
        self.gui.disconnect_optimizer(keep_timer = True)
        self.node.pos = self.gui.pixel_to_coor(pixel)
        self.gui.darea.queue_draw()
    def on_release(self, pixel):
        self.gui.tool = self.gui.basic_tool

class GeneralMoveOrRelease(Tool):
    def __init__(self, gui, pixel, tool_on_move, f_on_release, tolerance = 3):
        self.tolerance = 3
        self.pixel = pixel
        self.gui = gui
        self.tool_on_move = tool_on_move
        self.f_on_release = f_on_release
    def on_motion(self, pixel):
        if point_distance(pixel, self.pixel) > self.tolerance:
            self.gui.tool = self.tool_on_move(pixel)
    def on_release(self, pixel):
        self.gui.tool = self.gui.basic_tool
        self.f_on_release()

class MoveEdgeTool(Tool):
    def __init__(self, gui, edge, pixel = None):
        self.gui = gui
        self.edge = edge
        self.A, self.B = edge.endpoints
        if pixel is not None: self.on_motion(pixel)
    def on_motion(self, pixel):
        self.gui.disconnect_optimizer(keep_timer = True)
        coor = self.gui.pixel_to_coor(pixel)
        self.edge.curvature = arc_from_point(self.A, self.B, coor)
        self.gui.darea.queue_draw()
    def on_release(self, pixel):
        self.gui.tool = self.gui.basic_tool    

class ConstructSplit(Tool):
    def __init__(self, gui, node, pixel = None):
        self.gui = gui
        self.start_node = node
        self.path = []
        if pixel is not None: self.on_motion(pixel)
        else: self.pointer = node.pos
    def is_valid_edge(self, edge):
        return edge is not None and (bool(self.path) or edge not in self.start_node.edges)
    def on_motion(self, pixel):
        gui = self.gui
        node = gui.get_pixel_node(pixel)
        if node is not None and node != self.start_node:
            self.pointer = node.pos
        else:
            edge = gui.get_pixel_edge(pixel)
            if self.is_valid_edge(edge):
                A,B = [gui.coor_to_pixel(coor) for coor in edge.endpoints]
                P, subarcs = project_to_arc(A,B,edge.curvature, pixel)
                self.pointer = gui.pixel_to_coor(P)
            else:
                self.pointer = self.gui.pixel_to_coor(pixel)
        self.gui.darea.queue_draw()
    def on_left_click(self, pixel):
        gui = self.gui
        end_node = gui.get_pixel_node(pixel)
        if end_node is None:
            edge = gui.get_pixel_edge(pixel)
            if not self.is_valid_edge(edge):
                next_node = gui.pixel_to_coor(pixel)
                if not (self.path and self.path[-1] == next_node):
                    self.path.append(next_node)
                    self.gui.darea.queue_draw()
                return
            end_node = self.gui.bisect_edge_on_pixel(edge, pixel)
        if self.path: pos = self.path[0]
        else: pos = end_node.pos
        face_i = self.start_node.face_index_by_pos(pos)
        face = self.start_node.faces[face_i]
        classes = face.classes
        edge = face.split(self.start_node, end_node)
        if edge is None:
            print("cannot built this connection")
        else:
            gui.disconnect_optimizer(keep_timer = True)
            for f in edge.faces:
                f.set_classes(classes)
            if self.path:
                nodes = edge.multisect(len(self.path))
                for n,pos in zip(nodes, self.path):
                    n.pos = pos
        gui.tool = gui.basic_tool
        self.gui.darea.queue_draw()
    def on_release(self, pixel):
        pass
    def on_right_click(self, pixel):
        self.gui.tool = self.gui.basic_tool
        self.gui.darea.queue_draw()

    def display_fg(self, cr):
        gui = self.gui
        path = [self.start_node.pos] + self.path + [self.pointer]
        pixel_path = [gui.coor_to_pixel(coor) for coor in path]

        cr.set_line_width(3)
        cr.set_source_rgb(0,0.5,0)
        cr.move_to(*pixel_path[0])
        for pixel in pixel_path[1:]: cr.line_to(*pixel)
        cr.stroke()
        for pixel in pixel_path[1:-1]:
            x,y = pixel
            cr.arc(x, y, 5, 0, 2*math.pi)
            cr.fill()

class ReorderLabels(Tool):
    def __init__(self, gui, dragged):
        self.gui = gui
        self.dragged = dragged
    def on_release(self, point):
        self.gui.tool = self.gui.basic_tool
        self.gui.darea.queue_draw()
    def on_motion(self, pixel):
        cl = self.gui.legend.pixel_to_class(pixel, require_in = False)
        if cl is None:
            self.gui.tool = self.gui.basic_tool
            return
        if cl == self.dragged: return
        if cl > self.dragged:
            self.gui.graph.swap_class_labels(self.dragged, self.dragged+1)
            self.dragged += 1
        elif cl < self.dragged:
            self.gui.graph.swap_class_labels(self.dragged, self.dragged-1)
            self.dragged -= 1
        self.gui.darea.queue_draw()
