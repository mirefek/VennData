import math
import sys
import json
import itertools
from collections import defaultdict
from arc_geometry import *

def cyclic_shift(arr): return arr[1:]+[arr[0]]
def cyclic_pairs(arr): return zip(arr, cyclic_shift(arr))
def cyclic_triples(arr):
    arr2 = cyclic_shift(arr)
    arr3 = cyclic_shift(arr2)
    return zip(arr,arr2,arr3)

def count_angle_cw(angles): # how many times the angle values turned around
    return sum(
        int((a2 - a1 + math.pi) // (2*math.pi))
        for a1,a2 in cyclic_pairs(angles)
    )    

class Node:
    def __init__(self, graph):
        self.graph = graph
        self.graph.nodes.add(self)
        self.edges = []
        self.data = dict()
        # in case of an isolated node, it still has a single face associated to it

    @property
    def degree(self):
        return len(self.edges)
    @property
    def nodes(self):
        return [edge.other_node(self) for edge in self.edges]
    @property
    def faces(self):
        return [edge.cw_face(self) for edge in self.edges]
    @property
    def angles(self):
        res = []
        for edge in self.edges:
            p0,p1,curvature = edge.geometry_from(self)
            res.append(vector_direction(p0, p1) - curvature)
        return res
    @property
    def pos(self):
        return self.data['pos']
    @pos.setter
    def pos(self, pos):
        if self.data is None: self.data = dict()
        self.data['pos'] = pos

    def disconnect(self):
        self.graph.nodes.remove(self)

    def face_index_by_pos(self, pos):
        if len(self.edges) < 2: return 0
        main_angle = vector_direction(self.pos, pos)
        for i,(a1,a2) in enumerate(cyclic_pairs(self.angles)):
            if (main_angle-a2)%(2*math.pi) <= (a1-a2)%(2*math.pi): return i

        return None # comething is weird, normally should not happen, maybe with some weird rounding error

    def find_wrong_edge_order(self):
        if len(self.edges) <= 2: return None
        angles = self.angles
        for e,(a0,a1,a2) in zip(self.edges, cyclic_triples(angles)):
            if (a0-a2)%(2*math.pi) < (a1-a2)%(2*math.pi):
                return e
        return None

    def get_edges(self, other):
        return [
            edge for edge in self.edges
            if other in edge.nodes
        ]
    
    def unbisect(self):
        if len(self.edges) != 2: return None
        e0,e1 = self.edges
        f0,f1 = self.faces
        n0 = e0.other_node(self)
        n1 = e1.other_node(self)
        if n0 == n1: return None
        edge = Edge(n0,f0,n1,f1)

        # reconnect nodes
        n0.edges[n0.edges.index(e0)] = edge
        n1.edges[n1.edges.index(e1)] = edge

        # reconnect faces
        i = f0.edges.index(e1)
        if i < len(f0.edges)-1:
            assert f0.edges[i+1] == e0
            f0.edges[i:i+2] = [edge]
        else:
            assert f0.edges[0] == e0
            f0.edges[0] = edge
            f0.edges.pop()

        i = f1.edges.index(e0)
        if i < len(f1.edges)-1:
            assert f1.edges[i+1] == e1
            f1.edges[i:i+2] = [edge]
        else:
            assert f1.edges[0] == e1
            f1.edges[0] = edge
            f1.edges.pop()

        self.disconnect()
        e0.disconnect()
        e1.disconnect()
        return edge

class Edge:
    def __init__(self, n0,f0,n1,f1):
        self.graph = n0.graph
        self.graph.edges.add(self)
        self.nodes = [n0,n1]
        self.faces = [f0,f1]
        assert all(isinstance(n, Node) for n in self.nodes)
        assert all(isinstance(f, Face) for f in self.faces)
        self.data = dict()
        assert f0 != f1
        assert n0 != n1
        # cyclic order CW: node0, face0, node1, face1

    def disconnect(self):
        self.graph.edges.remove(self)

    @property
    def curvature(self):
        return self.data.get("curvature", 0.0)
    @curvature.setter
    def curvature(self, curvature):
        self.data["curvature"] = curvature

    @property
    def endpoints(self):
        return [node.pos for node in self.nodes]
    def length(self, m = math):
        direct = point_distance(*self.endpoints, m = m)
        return direct*arc_length(self.curvature, m = m)
    def area_from(self, face, m = math):
        A,B = self.endpoints
        res = arc_area(self.curvature, m = m) * point_distance_sq(A,B)
        if self.faces.index(face) == 1: return -res
        else: return res

    def other_node(self, node):
        i = self.nodes.index(node)
        return self.nodes[1-i]
    def other_face(self, face):
        i = self.faces.index(face)
        return self.faces[1-i]
    def cw_face(self, node):
        i = self.nodes.index(node)
        return self.faces[1-i]
    def ccw_face(self, node):
        i = self.nodes.index(node)
        return self.faces[i]
    def cw_node(self, face):
        i = self.faces.index(face)
        return self.nodes[i]
    def ccw_node(self, face):
        i = self.faces.index(face)
        return self.nodes[1-i]
    def _geometry_by_index(self, index):
        p0,p1 = self.endpoints
        if index == 0:
            return p0,p1, self.curvature
        else:
            return p1,p0, -self.curvature
    def geometry_cw(self,face):
        return self._geometry_by_index(1-self.faces.index(face))
    def geometry_ccw(self,face):
        return self._geometry_by_index(self.faces.index(face))
    def geometry_from(self,node):
        return self._geometry_by_index(self.nodes.index(node))

    def bisect(self):
        mid = Node(self.graph)
        n0,n1 = self.nodes
        f0,f1 = self.faces
        e0 = Edge(n0,f0,mid,f1)
        e1 = Edge(mid,f0,n1,f1)

        # reconnect nodes
        n0.edges[n0.edges.index(self)] = e0
        n1.edges[n1.edges.index(self)] = e1
        mid.edges.extend((e0,e1))

        # reconnect faces
        f0,f1 = self.faces
        i = f0.edges.index(self)
        f0.edges[i:i+1] = [e1,e0]
        i = f1.edges.index(self)
        f1.edges[i:i+1] = [e0,e1]

        self.disconnect()
        return mid

    def multisect(self, n):
        edge = self
        nodes = []
        for _ in range(n):
            node = edge.bisect()
            nodes.append(node)
            edge = node.edges[1]
        return nodes

    def join(self):
        n0,n1 = self.nodes
        f0,f1 = self.faces
        # check whether there is no other shared edge between the two faces
        all_edges_set = set(f0.edges)
        all_edges_set.update(f1.edges)
        if len(all_edges_set) != len(f0.edges)+len(f1.edges)-1:
            return None

        # disconnect nodes
        n0.edges.remove(self)
        n1.edges.remove(self)

        # create common face & connect it
        i0 = f0.edges.index(self)
        i1 = f1.edges.index(self)
        face = Face(self.graph)
        face.edges = f0.edges[i0+1:]+f0.edges[:i0]+f1.edges[i1+1:]+f1.edges[:i1]

        # reconnect edges
        for e in f0.edges:
            if e != self: e.faces[e.faces.index(f0)] = face
        for e in f1.edges:
            if e != self: e.faces[e.faces.index(f1)] = face

        # remove redundant elements
        self.disconnect()
        f0.disconnect()
        f1.disconnect()
        
        return face

def triangle_area(A,B,C):
    ax,ay = A
    bx,by = B
    cx,cy = C
    bx = bx-ax
    by = by-ay
    cx = cx-ax
    cy = cy-ay
    return (cx*by - bx*cy) / 2

class Face:
    def __init__(self, graph):
        self.graph = graph
        self.graph.faces.add(self)
        self.edges = []
        self.classes = frozenset()
        self.data = dict()
        # cyclic order CW: edge0, node0, face1, node1
        # except outer edge, where it is CCW

    def disconnect(self):
        self._disconnect_from_classes()
        self.graph.faces.remove(self)

    @property
    def faces(self):
        return [edge.other_face(self) for edge in self.edges]
    @property
    def nodes(self):
        return [edge.cw_node(self) for edge in self.edges]

    def _disconnect_from_classes(self):
        if not self.classes: return
        self.graph.classes_to_faces[self.classes].remove(self)
    def _connect_to_classes(self):
        if not self.classes: return
        self.graph.classes_to_faces[self.classes].add(self)

    def add_class(self, c):
        if c in self.classes: return
        self._disconnect_from_classes()
        self.classes = self.classes | frozenset([c])
        self._connect_to_classes()
    def remove_class(self, c):
        if c not in self.classes: return
        self._disconnect_from_classes()
        self.classes = self.classes - frozenset([c])
        self._connect_to_classes()
    def set_classes(self, classes):
        self._disconnect_from_classes()
        self.classes = frozenset(classes)
        self._connect_to_classes()

    def get_edges(self, other):
        return [
            edge for edge in self.edges
            if other in edge.faces
        ]

    def winding_number_cw(self, pos):
        """ calculate how many times the face revolves around pos, 1 = CW, -1 = CCW
        """
        res = 0
        angles = [
            vector_direction(pos, edge.ccw_node(self).pos)
            for edge in self.edges
        ]
        for (a0,a1),edge in zip(cyclic_pairs(angles), self.edges):
            a1x = a0+arc_angle(*edge.geometry_cw(self), pos)
            assert abs((a1-a1x+1) % (2*math.pi) - 1) < epsilon
            res += int((a1-a1x+1) // (2*math.pi))
        return res

    def turning_number_cw(self):
        """ calculate how many times one has to walk around, 1 = CW, -1 = CCW
        """
        angles = []
        for edge in self.edges:
            p0,p1,curvature = edge.geometry_cw(self)

            main_dir = vector_direction(p0, p1)
            angles.extend([
                main_dir-curvature,
                main_dir,
                main_dir+curvature,
            ])

        res = count_angle_cw(angles)
        return res

    def contains_pos(self, pos):
        winding = self.winding_number_cw(pos)
        turning = self.turning_number_cw()
        return winding >= max(0, turning)

    def split(self, n0, n1):
        if n0 == n1: return None
        nodes = self.nodes
        if n0 not in nodes or n1 not in nodes: return None
        i0 = nodes.index(n0)
        i1 = nodes.index(n1)
        f0 = Face(self.graph)
        f1 = Face(self.graph)
        edge = Edge(n0,f0,n1,f1)
        if i0 > i1:
            i0,i1 = i1,i0
            n0,n1 = n1,n0
            f0,f1 = f1,f0        

        # connect new faces
        f0.edges = self.edges[i0+1:i1+1]+[edge]
        f1.edges = self.edges[:i0+1]+[edge]+self.edges[i1+1:]

        # connect nodes
        n0.edges.insert(n0.faces.index(self)+1, edge)
        n1.edges.insert(n1.faces.index(self)+1, edge)

        # reconnect edges
        for e in f0.edges:
            if e != edge: e.faces[e.faces.index(self)] = f0
        for e in f1.edges:
            if e != edge: e.faces[e.faces.index(self)] = f1

        self.disconnect()

        return edge

    def area(self, m = math): # oriented area
        pos = [n.pos for n in self.nodes]
        A = pos[0]
        polygon_area = sum(triangle_area(A,B,C) for B,C in zip(pos[1:-1], pos[2:]))
        arc_area = sum(
            edge.area_from(self, m=m)
            for edge in self.edges
        )
        return polygon_area + arc_area

class PlanarGraph:
    def __init__(self):
        self.reset()

    def init_triangle(self):
        self.reset()
        outer_face = Face(self)
        inner_face = Face(self)
        nodes = [Node(self) for _ in range(3)]
        for n, pos in zip(nodes, [(0,1), (1,-1), (-1,-1)]):
            n.pos = pos
        edges = [
            Edge(n1, outer_face, n2, inner_face)
            for n1,n2 in cyclic_pairs(nodes)
        ]
        inner_face.edges.extend(edges)
        outer_face.edges.extend(reversed(edges))
        for node, edge_pair in zip(cyclic_shift(nodes), cyclic_pairs(edges)):
            node.edges.extend(edge_pair)

        n = nodes[0]
        i = n.face_index_by_pos((0,0))

    def dump_to_file(self, fname):
        with open(fname, 'w') as f:
            self.dump_to_stream(f)
    def print_self(self):
        self.dump_to_stream(sys.stdout)
    def dump_to_stream(self, stream):
        obj_to_name = dict()
        for classes, goal_area in self.classes_to_goal_areas.items():
            classes_str = '_'.join(str(x) for x in sorted(classes))
            print("goal_"+classes_str, ':', goal_area, file = stream)
        nodes = list(self.nodes)
        edges = list(self.edges)
        faces = list(self.faces)
        for i,n in enumerate(nodes): obj_to_name[n] = 'n'+str(i)
        for i,f in enumerate(faces): obj_to_name[f] = 'f'+str(i)
        for i,e in enumerate(edges): obj_to_name[e] = 'e'+str(i)
        for obj in nodes+faces+edges:
            if isinstance(obj, Edge):
                around = [obj.nodes[0], obj.faces[0], obj.nodes[1], obj.faces[1]]
            else: around = obj.edges
            name = obj_to_name[obj]
            print(name,':',*(obj_to_name[x] for x in around), file = stream)
            if obj.data:
                print(name+"_data", ':', json.dumps(obj.data), file = stream)
            if isinstance(obj, Face) and obj.classes:
                print(name+"_classes", ':', ' '.join(map(str, sorted(obj.classes))), file = stream)
        if self.metadata:
            print("metadata", ':', json.dumps(self.metadata), file = stream)

    def reset(self):
        self.nodes = set()
        self.faces = set()
        self.edges = set()
        self.classes_to_faces = defaultdict(set)
        self.classes_to_goal_areas = dict()
        self.metadata = dict()

    def load_from_file(self, fname):
        self.reset()

        name_content = []
        name_data = []
        name_classes = []
        name_to_obj = dict()
        with open(fname) as f:
            for line in f:
                name, content = line.split(' : ', 1)
                if name == "metadata":
                    self.metadata = json.loads(content)
                elif name.startswith("goal_"):
                    _, *classes = name.split('_')
                    classes = frozenset(int(x) for x in classes)
                    self.classes_to_goal_areas[classes] = float(content)
                elif name.endswith("_data"):
                    name_data.append((name[:-5], json.loads(content)))
                elif name.endswith("_classes"):
                    name_classes.append((name[:-8], frozenset(map(int, content.split()))))
                else:
                    name_content.append((name, content.split()))
        for name,_ in name_content:
            if name.startswith('n'):
                name_to_obj[name] = Node(self)
            elif name.startswith('f'):
                name_to_obj[name] = Face(self)
        for name, content in name_content:
            if name.startswith('e'):
                name_to_obj[name] = Edge(*(name_to_obj[x] for x in content))
        for name, content in name_content:
            if not name.startswith('e'):
                name_to_obj[name].edges = [name_to_obj[x] for x in content]
        for name, classes in name_classes:
            name_to_obj[name].set_classes(classes)
        for name, data in name_data:
            obj = name_to_obj[name]
            if isinstance(obj, Face):
                if "classes" in data:
                    classes = data.pop("classes")
                    obj.set_classes(classes)
                if "goal_area" in data:
                    goal_area = data.pop("goal_area")
                    self.classes_to_goal_areas.setdefault(obj.classes, 0)
                    self.classes_to_goal_areas[obj.classes] += abs(goal_area)
            obj.data = data

    def all_classes(self):
        all_classes = set()
        for classes, faces in self.classes_to_faces.items():
            if not faces: continue
            all_classes.update(classes)
        return sorted(all_classes)

    def set_goal_areas(self, loaded_data):
        data = {
            frozenset(item['classes']) : item['size']
            for item in loaded_data['data']
        }
        total_current_area = 0
        total_data_size = sum([
            size
            for classes, size in data.items()
            if classes
        ])
        total_current_area = sum([
            face.area()
            for face in self.faces
            if face.classes
        ])
        coef = total_current_area / total_data_size
        print(total_current_area)
        self.classes_to_goal_areas = {
            classes : size*coef
            for classes, size in data.items()
            if classes
        }
        empty_space = data.get(frozenset(), None)
        if empty_space is not None:
            self.metadata["total_area"] = coef*(total_data_size + empty_space)
            self.metadata["class_labels"] = loaded_data["class_labels"]

    @staticmethod
    def merged_faces(faces):
        edge_count = dict()
        for face in set(faces):
            for edge in face.edges:
                cnt = 2*edge.index(face)-1
                if edge in edge_count:
                    edge_count[edge] += cnt
                else:
                    edge_count[edge] = cnt
        node_to_next = dict()
        for edge, cnt in edge_count.items():
            assert abs(cnt) <= 1
            n0,n1 = edge.nodes
            if cnt > 0: node_to_next[n0] = n1,e
            if cnt < 0: node_to_next[n1] = n0,e

        components = []
        while node_to_next:
            start_node = next(node_to_next.keys())
            node = start_node
            component = []
            while True:
                node,e = node_to_next.pop(node)
                component.append((node, e))
                if node == start_node: break
            components.append(component)
        return components

    def class_nodes(self, class_index):
        faces = [
            face for face in self.faces
            if class_index in face.classes
        ]
        return self.merged_faces(faces)

    def get_classes_area(self, classes, m = math):
        return sum(face.area(m = m) for face in self.classes_to_faces[classes])

    def bounding_box(self):
        right,up,left,down = (
            max(
                max_in_direction(*edge.endpoints, edge.curvature, direction)
                for edge in self.edges
            )
            for direction in [(1,0), (0,1), (-1,0), (0,-1)]
        )
        return (-left,up), (left+right,up+down)

    def swap_class_labels(self, cl1, cl2):
        i1 = cl1-1
        i2 = cl2-1
        labels = self.metadata["class_labels"]
        labels[i1], labels[i2] = labels[i2], labels[i1]

        cl12 = frozenset([cl1,cl2])
        def modify_classes(classes):
            cl1_in = cl1 in classes
            cl2_in = cl2 in classes
            if cl1_in == cl2_in: return classes
            else: return cl12 ^ classes
        self.classes_to_goal_areas = {
            modify_classes(classes) : goal_area
            for classes, goal_area in self.classes_to_goal_areas.items()
        }
