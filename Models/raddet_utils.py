import torch

def iou2d(box_xywh_1, box_xywh_2):
    """
    2D IoU calculation for bounding boxes
    Args:
        box_xywh_1        ->      box1 [x, y, w, h]
        box_xywh_2        ->      box2 [x, y, w, h]
        input_shape       ->      optional input shape for compatibility
    """
    assert box_xywh_1.shape[-1] == 4
    assert box_xywh_2.shape[-1] == 4
    
    device = box_xywh_1.device
    
    ### areas of both boxes
    box1_area = box_xywh_1[..., 2] * box_xywh_1[..., 3]
    box2_area = box_xywh_2[..., 2] * box_xywh_2[..., 3]
    
    ### find the intersection box
    box1_min = box_xywh_1[..., :2] - box_xywh_1[..., 2:] * 0.5
    box1_max = box_xywh_1[..., :2] + box_xywh_1[..., 2:] * 0.5
    box2_min = box_xywh_2[..., :2] - box_xywh_2[..., 2:] * 0.5
    box2_max = box_xywh_2[..., :2] + box_xywh_2[..., 2:] * 0.5

    top_left = torch.maximum(box1_min, box2_min)
    bottom_right = torch.minimum(box1_max, box2_max)
    
    ### get intersection area
    intersection = torch.maximum(bottom_right - top_left, torch.tensor(0.0, device=device))
    intersection_area = intersection[..., 0] * intersection[..., 1]
    
    ### get union area
    union_area = box1_area + box2_area - intersection_area
    
    ### get iou
    iou = torch.nan_to_num(intersection_area / (union_area + 1e-10))
    return iou
