import torch
from arc_geometry import *
from collections import defaultdict

class PlanarGraphOptimizer:
    def __init__(self, default_step = 0.01, default_coef = 5.0):
        self.connected = False
        self.default_step = default_step
        self.default_coef = default_coef

    def connect(self, graph):
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.edges = list(graph.edges)
        learnable = []
        for node in self.nodes:
            learnable.extend(node.pos)
        for edge in self.edges:
            A,B = edge.endpoints
            learnable.append(math.tan(edge.curvature/2)*point_distance(A,B))
        self.learnable = torch.as_tensor(learnable, dtype = torch.float64)
        self.learnable.requires_grad = True
        self.connected = True
        self.put_learnable_in_place()

    def put_learnable_in_place(self):
        for i,node in enumerate(self.nodes):
            node.pos = self.learnable[2*i:2*(i+1)]
        for i,edge in enumerate(self.edges):
            A,B = edge.endpoints
            dist = point_distance(A,B, m = torch)
            edge.curvature = torch.atan(self.learnable[i+len(self.nodes)*2]/dist)*2

    def disconnect(self):
        if not self.connected: return
        for node in self.nodes:
            x,y = node.pos
            x = float(x)
            y = float(y)
            node.pos = (x,y)
        for edge in self.edges:
            edge.curvature = float(edge.curvature)
        del self.graph
        del self.nodes
        del self.learnable
        self.connected = False

    def f_to_gradient(self, loss):
        self.put_learnable_in_place()
        self.learnable.grad = None
        value = loss()
        value.backward()
        res = self.learnable.grad
        self.learnable.grad = None
        return res, value.item()

    def area_MSE(self):
        res = 0.0
        for classes, goal_area in self.graph.classes_to_goal_areas.items():
            current_area = self.graph.get_classes_area(classes, m = torch)
            if not isinstance(current_area, torch.Tensor): continue
            res = res + (current_area - goal_area)**2
        return res

    def total_length(self):
        res = 0.0
        for edge in self.graph.edges:
            f0,f1 = edge.faces
            classes = f0.classes ^ f1.classes
            if not classes: continue
            res += len(classes)*edge.length(m = torch)
        return res

    def smoothness_loss(self):
        loss = torch.as_tensor(0.0)
        for node in self.nodes:
            classes_to_angles = defaultdict(list)
            for edge, angle in zip(node.edges, node.angles(m = torch)):
                classes_to_angles[edge.classes].append(angle)
            for angles in classes_to_angles.values():
                if len(angles) != 2: continue
                a0,a1 = angles
                cur_loss = ((a0-a1) % (2*math.pi) - math.pi)**2
                loss = loss + cur_loss
        return loss

    def get_numerical_gradient(self, f, step = 0.01):
        res = []
        main_value = float(f())
        for node in self.nodes:
            ori = node.pos
            x,y = ori
            node.pos = (x+step, y)
            res.append((float(f())-main_value)/step)
            node.pos = (x, y+step)
            res.append((float(f())-main_value)/step)
            node.pos = ori
        for edge in self.edges:
            ori = edge.curvature
            edge.curvature = ori+step
            res.append((float(f())-main_value)/step)
            edge.curvature = ori
        return res

    def check_numerical_gradients(self):
        
        self.learnable.grad = None
        print("=== numerical gradients ===")
        print("Areas:")
        for face in self.graph.faces:
            print(self.get_numerical_gradient(face.area))
        print("Total Length:")
        print(self.get_numerical_gradient(self.total_length))
        print("Custom:")
        print(self.get_numerical_gradient(self.area_MSE))
        #print(self.get_numerical_gradient(lambda: self.faces[0].area(m = torch)))
        #print(self.get_numerical_gradient(lambda: self.edges[0].area_from(self.faces[0], m = torch)))
        #print(self.get_numerical_gradient(lambda: arc_area(self.edges[0].curvature, m = torch)))

    def project_gradient(self, grad):
        base = []

        def orthogonal_to_base(v):
            for b in base:
                v = v - b*torch.dot(b.flatten(), v.flatten())
            return v

        for classes, goal_area in self.graph.classes_to_goal_areas.items():
            if not self.graph.classes_to_faces.get(classes,None): continue
            v,_ = self.f_to_gradient(lambda: self.graph.get_classes_area(classes, m = torch))
            v = orthogonal_to_base(v)
            norm = torch.norm(v)
            if norm < 0.0001: continue
            v = v / norm
            base.append(v)

        res = orthogonal_to_base(grad)
        return res

    def calculate_gradient(self, coef, loss_f = None):
        area_MSE_grad,area_loss = self.f_to_gradient(self.area_MSE)
        if loss_f is None:
            print("area_loss:", area_loss)
            return area_MSE_grad
        total_length_grad,total_length = self.f_to_gradient(loss_f)
        total_length_grad = self.project_gradient(total_length_grad)
        print("area_loss:", area_loss, "total_length:", total_length)
        
        return area_MSE_grad + coef*total_length_grad
    
    def gradient_step(self, grad = None, step = None, coef = None):
        if step is None: step = self.default_step
        if coef is None: coef = self.default_coef
        if not isinstance(grad, torch.Tensor):
            grad = self.calculate_gradient(coef, grad)
        with torch.no_grad(): self.learnable -= step*grad
        self.put_learnable_in_place()

    def gradient_step_min_len(self, *args, **kwargs):
        self.gradient_step(self.total_length, *args, **kwargs)

    def gradient_step_smooth(self, *args, **kwargs):
        self.gradient_step(self.smoothness_loss, *args, **kwargs)
