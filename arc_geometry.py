import math

epsilon = 0.0001

def vector_direction(A,B, m = math):
    ax,ay = A
    bx,by = B
    return m.atan2(by-ay,bx-ax)
def point_distance_sq(A,B):
    ax,ay = A
    bx,by = B
    return (ax-bx)**2 + (ay-by)**2
def point_distance(A,B, m = math):
    ax,ay = A
    bx,by = B
    return m.sqrt((ax-bx)**2 + (ay-by)**2)

def normalize_angle(a):
    return (a + math.pi)%(2*math.pi) - math.pi

# assuming an arc from (0,0) to (0,1), represented by tangent angle
# tangent angle is
# * 0 for the case of straight line,
# * negative from (-pi, 0) if the arc is in the left half-plane
# * positive from (0, pi) if the arc is in the right half-plane

def arc_from_point(A,B, point):
    return normalize_angle(vector_direction(point,B) - vector_direction(A,point))

def arc_length(arc, m = math):
    if abs(arc) < epsilon: return 1
    else: return abs(arc / m.sin(arc))

def arc_area(arc, m = math): # positive if on the right, scaled to length 1
    if abs(arc) < epsilon: return arc/3
    else:
        # most cumputation with arc length 2
        radius = 1 / (2*m.sin(arc))
        rx = m.tan(arc + math.pi/2)
        tri_area = -rx/4
        return radius**2 * arc - tri_area    

def arc_center(A,B,arc):
    ax,ay = A
    bx,by = B
    mx = (ax+bx)/2
    my = (ay+by)/2
    vx = my-ay
    vy = ax-mx
    rc = math.tan(arc + math.pi/2)
    rx = mx+rc*vx
    ry = my+rc*vy
    return rx,ry

def arc_angle(A,B,arc, point): # CCW angle from A to B through arc as seen from point, with the correct sign
    res = vector_direction(point,B) - vector_direction(point,A)
    if abs(arc) < epsilon:
        return normalize_angle(res)
    else:
        x,y = point
        ax,ay = A
        bx,by = B
        mx = (ax+bx)/2
        my = (ay+by)/2
        vx = my-ay
        vy = ax-mx
        if (vx*(x-ax) + vy*(y-ay))*arc < -epsilon:
            return normalize_angle(res)
        rc = math.tan(arc + math.pi/2)
        rx = mx+rc*vx
        ry = my+rc*vy
        radius = point_distance(A,(rx,ry))
        dist = point_distance(point,(rx,ry))
        if dist > radius:
            return normalize_angle(res)
        elif arc > 0:
            return (res%(2*math.pi))
        else:
            return -((-res)%(2*math.pi))

def max_in_direction(A,B,arc,D): # D is a normalized vector
    ax,ay = A
    bx,by = B
    dx,dy = D
    value_a = ax*dx + ay*dy
    value_b = bx*dx + by*dy
    da = math.atan2(dy,dx)
    narc_d = vector_direction(A,B)+math.pi/2
    if arc > 0: narc_d = narc_d + math.pi
    if abs(normalize_angle(narc_d - da)) > abs(arc) or abs(arc) < epsilon:
        return max(value_a, value_b)
    else:
        center = arc_center(A,B,arc)
        cx,cy = center
        radius = point_distance(center,A)
        value_c = cx*dx + cy*dy + radius
        return max(value_a, value_b, value_c)

# expects (revesed) cairo coordinates
def _arc_drawing_data(A,B,arc):
    ax,ay = A
    bx,by = B
    mx = (ax+bx)/2
    my = (ay+by)/2
    vx = ay-my
    vy = mx-ax
    rc = math.tan(arc + math.pi/2)

    rx = mx+rc*vx
    ry = my+rc*vy
    radius = abs(point_distance(A,B)/(2*math.sin(arc)))
    segment_direction = vector_direction(A,B)
    if arc < 0:
        a0 = arc-math.pi/2 + segment_direction
        a1 = -arc-math.pi/2 + segment_direction
        return rx,ry,radius,a0,a1, True
    else:
        a0 = arc+math.pi/2 + segment_direction
        a1 = -arc+math.pi/2 + segment_direction
        return rx,ry,radius,a0,a1, False

# expects (revesed) cairo coordinates,
# returns None if the closest is an endpoint
# otherwise, returns (P, (arc0, arc1))
# where P are the coordinates of the projection
# and arc0, arc1 are the tangent angles for the two parts
def project_to_arc(A,B, arc, point):

    if A == B: return None
    x,y = point
    if abs(arc) < epsilon: # projection to line
        ax,ay = A
        bx,by = B
        if (bx-ax)*(x-ax) + (by-ay)*(y-ay) < 0: return None,None # outside of the line
        if (ax-bx)*(x-bx) + (ay-by)*(y-by) < 0: return None,None
        nx,ny = (ay-by,bx-ax)
        dist = math.sqrt(nx**2 + ny**2)
        nx = nx / dist
        ny = ny / dist
        c = nx*(ax-x) + ny*(ay-y)
        res = (x + c*nx, y + c*ny)
        return res, (0,0)
    else:
        rx,ry,radius,a0,a1, clockwise = _arc_drawing_data(A,B,arc)
        if abs(rx-x)+abs(ry-y) < epsilon: return None,None # too close to the circle center
        ap = vector_direction((rx,ry), point)
        if ((ap-a0)%(2*math.pi) >= (a1-a0)%(2*math.pi)) == clockwise:
            return None,None # outside of the arc
        P = (rx+radius*math.cos(ap), ry+radius*math.sin(ap))
        if clockwise:
            arc0 = -((ap-a0) % (2*math.pi))
            arc1 = -((a1-ap) % (2*math.pi))
        else:
            arc0 = (a0-ap) % (2*math.pi)
            arc1 = (ap-a1) % (2*math.pi)
        return P, (arc0 / 2, arc1 / 2)

def draw_arc(cr, A,B, arc): # expects (revesed) cairo coordinates
    arc = float(arc)
    if abs(arc) < epsilon:
        cr.line_to(*B)
    else:
        rx,ry,radius,a0,a1, clockwise = _arc_drawing_data(A,B,arc)
        if clockwise: cr.arc(rx,ry,radius,a0,a1)
        else: cr.arc_negative(rx,ry,radius,a0,a1)
