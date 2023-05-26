from planar_graph import *

class GraphCursor:
    def __init__(self, gui):
        self.gui = gui
        self.graph = self.gui.graph
        self.edge = None
        self.node_or_face = None
        
    def assure_valid(self):
        if self.disabled: return
        if self.edge not in self.node_or_face.edges:
            self.disable()
        elif isinstance(self.node_or_face, Node):
            if self.node_or_face not in self.graph.nodes:
                self.disable()
        elif isinstance(self.node_or_face, Face):
            if self.node_or_face not in self.graph.faces:
                self.disable()

    @property
    def disabled(self):
        return self.edge is None
    def enable(self):
        self.node_or_face = next(iter(self.graph.faces))
        self.edge = self.node_or_face.edges[0]
    def disable(self):
        self.node_or_face = None
        self.edge = None

    @property
    def node(self):
        if isinstance(self.node_or_face, Node): return self.node_or_face
        else: return None
    @property
    def face(self):
        if isinstance(self.node_or_face, Face): return self.node_or_face
        else: return None

    def move_around_obj(self, step):
        self.assure_valid()
        if self.disabled: self.enable()
        index = self.node_or_face.edges.index(self.edge)
        index = (index+step) % len(self.node_or_face.edges)
        self.edge = self.node_or_face.edges[index]
    def move_around_edge(self, step):
        self.assure_valid()
        if self.disabled: self.enable()
        n0,n1 = self.edge.nodes
        f0,f1 = self.edge.faces
        objs = [n0,f0,n1,f1]
        index = objs.index(self.node_or_face)
        self.node_or_face = objs[(index+step)%4]

    def display_bg(self, cr):
        self.assure_valid()
        if isinstance(self.node_or_face, Face):
            cr.set_source_rgba(1, 0, 1, 0.2)
            self.gui.draw_face(cr, self.node_or_face)

    def display_fg(self, cr):
        if self.edge is not None:
            cr.set_source_rgb(0.5,0,0.5)
            cr.set_line_width(6)
            cr.set_source_rgb(0.5, 0, 0.5)
            self.gui.draw_edge(cr, self.edge)
        if isinstance(self.node_or_face, Node):
            cr.set_source_rgb(1, 0, 1)
            self.gui.draw_node(cr, self.node_or_face, size = 5)

    def move_by_key(self, keyval_name):
        if keyval_name == 'Up':
            self.move_around_edge(2)
        elif keyval_name == 'Left':
            self.move_around_obj(-1)
        elif keyval_name == 'Right':
            self.move_around_obj(1)
        elif keyval_name == 'period':
            self.move_around_edge(-1)
        elif keyval_name == 'slash':
            self.move_around_edge(1)
        else: return False

        return True
