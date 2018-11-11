import os
import pandas as pd
import numpy as np
import ast
import build_drawings

data = pd.read_csv('train_simplified/fence.csv')
data = data.iloc[0,:]
data = data['drawing']
data = ast.literal_eval(data)
data = build_drawings.build_drawing(data,256)

def find_non_zero_nodes(data):
    (row_ind, col_ind) = np.where(data)
    return col_ind, row_ind

def get_up_node_coords(x_coords,y_coords):
    return x_coords,y_coords-1

def get_down_node_coords(x_coords,y_coords):
    return x_coords,y_coords+1

def get_left_node_coords(x_coords,y_coords):
    return x_coords-1,y_coords

def get_right_node_coords(x_coords,y_coords):
    return x_coords+1,y_coords

def get_up_left_node_coords(x_coords,y_coords):
    return x_coords-1,y_coords-1

def get_up_right_node_coords(x_coords,y_coords):
    return x_coords+1,y_coords-1

def get_down_left_node_coords(x_coords,y_coords):
    return x_coords-1,y_coords+1

def get_down_right_node_coords(x_coords,y_coords):
    return x_coords+1,y_coords+1

def get_valid_coord(x_coord, y_coord, size):
    valid_x_coord = np.logical_and(x_coord>=0, x_coord<size)
    valid_y_coord = np.logical_and(y_coord>=0, y_coord<size)
    valid_coord = np.logical_and(valid_x_coord, valid_y_coord)
    return valid_coord
    #return x_coord[valid_coord], y_coord[valid_coord], size

def intersect_coords(x_coord, y_coord, x_coord_ngbr, y_coord_ngbr):
    comm_x = np.in1d(x_coord, x_coord_ngbr)
    comm_y = np.in1d(y_coord, y_coord_ngbr)
    return np.logical_and(comm_x, comm_y)

def sub2ind(x_coord, y_coord, size):
    return y_coord*size + x_coord

def set_h_edge_feature(x_coords, y_coords, size):
    tot_nodes = size*size
    h_edge_feature_tt = np.zeros(tot_nodes)
    h_edge_feature_tf = np.zeros(tot_nodes)
    h_edge_feature_ft = np.zeros(tot_nodes)

    x_coords_left, y_coords_left = get_left_node_coords(x_coords,y_coords)
    x_coords_right, y_coords_right = get_right_node_coords(x_coords,y_coords)

    valid_ngbr = get_valid_coord(x_coords_left, y_coords_left, size)
    true_ngbr = intersect_coords(x_coords,y_coords,x_coords_left,y_coords_left)
    inds = np.logical_and(true_ngbr, valid_ngbr)
    inds = sub2ind(x_coords_left[inds], y_coords_left[inds], size)
    h_edge_feature_tt[inds] = 1

    false_ngbr = np.logical_not(true_ngbr)
    inds = np.logical_and(false_ngbr, valid_ngbr)
    inds = sub2ind(x_coords_left[inds], y_coords_left[inds], size)
    h_edge_feature_ft[inds] = 1

    valid_ngbr = get_valid_coord(x_coords_right, y_coords_right, size)
    true_ngbr = intersect_coords(x_coords, y_coords, x_coords_right, y_coords_right)
    inds = np.logical_and(true_ngbr, valid_ngbr)
    inds = sub2ind(x_coords[inds], y_coords[inds], size)
    h_edge_feature_tt[inds] = 1

    false_ngbr = np.logical_not(true_ngbr)
    inds = np.logical_and(false_ngbr, valid_ngbr)
    inds = sub2ind(x_coords_right[inds], y_coords_right[inds], size)
    h_edge_feature_tf[inds] = 1

    return np.concatenate(h_edge_feature_tt, h_edge_feature_tf, h_edge_feature_ft)



def set_feature_mat(drawing, size):
    feature_node = np.zeros(size)
    (x_coords, y_coords) = find_non_zero_nodes(drawing)
    feature_node[sub2ind(x_coords, y_coords, size)] = 1
    feature_right_edge = set_h_edge_feature(x_coords, y_coords, size)


print(data)