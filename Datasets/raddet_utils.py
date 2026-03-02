import numpy as np

def smooth_onehot(class_id, num_classes, smooth_coef=0.01):
    """ Transfer class index to one hot class (smoothed) """
    assert isinstance(class_id, int)
    assert isinstance(num_classes, int)
    assert class_id < num_classes
    # building one hot vector
    onehot_vector = np.zeros(num_classes, dtype=np.float32)
    onehot_vector[class_id] = 1.0
    # smoothing
    uniform_vector = np.full(num_classes, 1.0 / num_classes, dtype=np.float32)
    smooth_vector = (1-smooth_coef) * onehot_vector + smooth_coef * uniform_vector
    return smooth_vector


def iou2d(box_xywh_1, box_xywh_2):
    """ Numpy version of 2D bounding box IOU calculation
    Args:
        box_xywh_1        ->      box1 [x, y, w, h]
        box_xywh_2        ->      box2 [x, y, w, h]
    """
    assert box_xywh_1.shape[-1] == 4
    assert box_xywh_2.shape[-1] == 4

    # areas of both boxes
    box1_area = box_xywh_1[..., 2] * box_xywh_1[..., 3]
    box2_area = box_xywh_2[..., 2] * box_xywh_2[..., 3]
    # find the intersection box
    box1_min = box_xywh_1[..., :2] - box_xywh_1[..., 2:] * 0.5
    box1_max = box_xywh_1[..., :2] + box_xywh_1[..., 2:] * 0.5
    box2_min = box_xywh_2[..., :2] - box_xywh_2[..., 2:] * 0.5
    box2_max = box_xywh_2[..., :2] + box_xywh_2[..., 2:] * 0.5
    top_left = np.maximum(box1_min, box2_min)
    bottom_right = np.minimum(box1_max, box2_max)
    
    # get intersection area
    intersection = np.maximum(bottom_right - top_left, 0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1]

    # get union area
    union_area = box1_area + box2_area - intersection_area
    
    # get iou
    iou = np.nan_to_num(intersection_area / (union_area + 1e-10))
    return iou


def sort_anchor_by_area(anchors):
    anchor_area_list = []
    for anchor in anchors:
        cur_area = 1.0
        for cur_dim in anchor:
            cur_area *= cur_dim
        anchor_area_list.append(cur_area)
    sorted_idx = np.argsort(anchor_area_list)
    return anchors[sorted_idx]

def read_anchors(anchor_fname):
    anchors = []
    with open(anchor_fname, 'r') as f:
        lines = f.readlines()
        for line in lines:
            box = []
            coord_list = line.strip().split(' ')
            for coord in coord_list:
                box.append(int(coord))
            anchors.append(box)
    anchors = np.array(anchors)
    sorted_anchors = sort_anchor_by_area(anchors)
    return sorted_anchors


