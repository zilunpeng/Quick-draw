import numpy as np

def plotLineLow(x0,y0,x1,y1):
    pt_x = []
    pt_y = []
    dx = x1 - x0
    dy = y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy
    D = 2*dy - dx
    y = y0
    for x in np.arange(x0,x1+1):
        pt_x.append(x), pt_y.append(y)
        if D>0:
            y = y+yi
            D = D-2*dx
        D = D+2*dy
    return pt_x, pt_y

def plotLineHigh(x0,y0,x1,y1):
    pt_x = []
    pt_y = []
    dx = x1 - x0
    dy = y1 - y0
    xi = 1
    if dx < 0:
        xi = -1
        dx = -dx
    D = 2*dx - dy
    x = x0
    for y in np.arange(y0,y1+1):
        pt_x.append(x), pt_y.append(y)
        if D>0:
            x = x+xi
            D = D-2*dy
        D = D+2*dx
    return pt_x, pt_y

def get_line_int_coords(x0,y0,x1,y1):
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            x_coords, y_coords = plotLineLow(x1, y1, x0, y0)
        else:
            x_coords, y_coords = plotLineLow(x0, y0, x1, y1)
    else:
        if y0 > y1:
            x_coords, y_coords = plotLineHigh(x1, y1, x0, y0)
        else:
            x_coords, y_coords = plotLineHigh(x0, y0, x1, y1)
    return x_coords, y_coords