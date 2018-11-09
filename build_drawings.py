import os
import pandas as pd
import numpy as np
import ast

num_img_to_show = 64
data = pd.read_csv('train_simplified/fence.csv')
num_imgs = data.shape[0]
data = data.iloc[np.random.choice(num_imgs,size=num_img_to_show,replace=False),:]
data['drawing'] = data['drawing'].apply(ast.literal_eval)
data = data['drawing']

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

def build_drawing(drawing,size):
    img = np.zeros((size,size))
    for stroke in drawing:
        for i in range(len(stroke[0]) - 1):
            x_coords, y_coords = get_line_int_coords(stroke[0][i], stroke[1][i], stroke[0][i + 1], stroke[1][i + 1])
            img[x_coords, y_coords] = 1
    return img

import matplotlib.pyplot as plt
n = int(np.sqrt(num_img_to_show))
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(12, 12))
for i in range(num_img_to_show):
    ax = axs[i // n, i % n]
    ax.imshow(build_drawing(data.iloc[i],256), cmap=plt.cm.gray)
    ax.axis('off')
    # plt.imshow(build_drawing(data.iloc[i],256), cmap=plt.cm.gray)
    # plt.axis('off')
    # plt.show()

plt.tight_layout()
fig.savefig('samples.png', dpi=300)
plt.show()