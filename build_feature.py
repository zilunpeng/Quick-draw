import pandas as pd
import numpy as np
import ast
import build_drawings
import scipy.sparse

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

def intersect_coords(x_coord, y_coord, x_coord_ngbr, y_coord_ngbr):
    comm_x = np.in1d(x_coord, x_coord_ngbr)
    comm_y = np.in1d(y_coord, y_coord_ngbr)
    return np.logical_and(comm_x, comm_y)

def sub2ind(x_coord, y_coord, size):
    return y_coord*size + x_coord

def get_tt_ind(x, y, x_ngbr, y_ngbr, is_ngbr_center, size):
    valid_ngbr = get_valid_coord(x_ngbr, y_ngbr, size)
    true_ngbr = intersect_coords(x, y, x_ngbr, y_ngbr)
    inds = np.logical_and(true_ngbr, valid_ngbr)
    inds = sub2ind(x_ngbr[inds], y_ngbr[inds], size) if is_ngbr_center else sub2ind(x[inds], y[inds], size)
    return inds, true_ngbr, valid_ngbr

def get_tf_ft_ind(x,y,x_ngbr,y_ngbr,true_ngbr, valid_ngbr, is_ngbr_center, size):
    false_ngbr = np.logical_not(true_ngbr)
    inds = np.logical_and(false_ngbr, valid_ngbr)
    inds = sub2ind(x_ngbr[inds], y_ngbr[inds], size) if is_ngbr_center else sub2ind(x[inds], y[inds], size)
    return inds

def set_edge_feature(x, y, get_center_ngbr_coords, get_noncent_ngbr_coords, size):
    tot_nodes = size*size
    edge_feature_tt = np.zeros(tot_nodes)
    edge_feature_tf = np.zeros(tot_nodes)
    edge_feature_ft = np.zeros(tot_nodes)
    x_cent, y_cent = get_center_ngbr_coords(x,y)
    x_noncent, y_noncent = get_noncent_ngbr_coords(x,y)

    inds, true_ngbr, valid_ngbr = get_tt_ind(x,y,x_cent,y_cent,is_ngbr_center=True,size=size)
    edge_feature_tt[inds] = 1
    edge_feature_ft[get_tf_ft_ind(x,y,x_cent,y_cent,true_ngbr,valid_ngbr,is_ngbr_center=True,size=size)] = 1

    inds, true_ngbr, valid_ngbr = get_tt_ind(x,y,x_noncent,y_noncent,is_ngbr_center=False,size=size)
    edge_feature_tt[inds] = 1
    edge_feature_tf[get_tf_ft_ind(x,y,x_noncent,y_noncent,true_ngbr,valid_ngbr,is_ngbr_center=False,size=size)] = 1

    return np.concatenate((edge_feature_tt, edge_feature_tf, edge_feature_ft))

def set_feature_mat(drawing, size):
    feature_node = np.zeros(size*size)
    (x, y) = find_non_zero_nodes(drawing)
    feature_node[sub2ind(x, y, size)] = 1
    feature_h_edge = set_edge_feature(x,y,get_left_node_coords,get_right_node_coords,size)
    feature_v_edge = set_edge_feature(x,y,get_up_node_coords,get_down_node_coords,size)
    feature_nw_edge = set_edge_feature(x,y,get_up_left_node_coords,get_down_right_node_coords,size)
    feature_ne_edge = set_edge_feature(x,y,get_up_right_node_coords,get_down_left_node_coords,size)
    return np.concatenate((feature_node,feature_h_edge,feature_v_edge,feature_nw_edge,feature_ne_edge))

if __name__ == "main":
    data = pd.read_csv('train_simplified/fence.csv')
    print('data size: ', data.shape)
    data = data.iloc[0,:]
    data = data['drawing']
    data = ast.literal_eval(data)
    data = build_drawings.build_drawing(data,256)
    feature = set_feature_mat(data,256)
# print('num non-zero: ', np.count_nonzero(feature))
# sp_mat = scipy.sparse.csc_matrix(feature)
# scipy.sparse.save_npz('sparse_feature_testing.npz',sp_mat)
# testing_sp_mat = scipy.sparse.load_npz('sparse_feature_testing.npz')
# print(testing_sp_mat)
