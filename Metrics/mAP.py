import torch


def iou2d(box_xywh_1, box_xywh_2):
    """
    Args:
        box_xywh_1        ->      box1 [x, y, w, h]
        box_xywh_2        ->      box2 [x, y, w, h]"""
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


def iou3d(box_xyzwhd_1, box_xyzwhd_2, input_shape):
    """
    Args:
        box_xyzwhd_1        ->      box1 [x, y, z, w, h, d]
        box_xyzwhd_2        ->      box2 [x, y, z, w, h, d]"""
    assert box_xyzwhd_1.shape[-1] == 6
    assert box_xyzwhd_2.shape[-1] == 6
    
    device = box_xyzwhd_1.device
    fft_shift_implement = torch.tensor([0, 0, input_shape[2] / 2], device=device)
    
    ### areas of both boxes
    box1_area = box_xyzwhd_1[..., 3] * box_xyzwhd_1[..., 4] * box_xyzwhd_1[..., 5]
    box2_area = box_xyzwhd_2[..., 3] * box_xyzwhd_2[..., 4] * box_xyzwhd_2[..., 5]
    
    ### find the intersection box
    box1_min = box_xyzwhd_1[..., :3] + fft_shift_implement - box_xyzwhd_1[..., 3:] * 0.5
    box1_max = box_xyzwhd_1[..., :3] + fft_shift_implement + box_xyzwhd_1[..., 3:] * 0.5
    box2_min = box_xyzwhd_2[..., :3] + fft_shift_implement - box_xyzwhd_2[..., 3:] * 0.5
    box2_max = box_xyzwhd_2[..., :3] + fft_shift_implement + box_xyzwhd_2[..., 3:] * 0.5

    top_left = torch.maximum(box1_min, box2_min)
    bottom_right = torch.minimum(box1_max, box2_max)
    
    ### get intersection area
    intersection = torch.maximum(bottom_right - top_left, torch.tensor(0.0, device=device))
    intersection_area = intersection[..., 0] * intersection[..., 1] * intersection[..., 2]
    
    ### get union area
    union_area = box1_area + box2_area - intersection_area
    
    ### get iou
    iou = torch.nan_to_num(intersection_area / (union_area + 1e-10))
    return iou


def get_tp_oneclass(pred, gt, input_shape, iou_threshold=0.5, mode="3D"):
    """ output tp (true positive) with size [num_pred, ] """
    assert mode in ["3D", "2D"]
    tp = torch.zeros(len(pred), device=pred.device)
    detected_gt_boxes = []
    for i in range(len(pred)):
        if len(detected_gt_boxes) == len(gt): 
            break
        
        current_pred = pred[i]
        if mode == "3D":
            current_pred_box = current_pred[:6]
            gt_box = gt[..., :6]
        else:
            current_pred_box = current_pred[:4]
            gt_box = gt[..., :4]

        if mode == "3D":
            iou = iou3d(current_pred_box.unsqueeze(0), gt_box, input_shape)
        else:
            iou = iou2d(current_pred_box.unsqueeze(0), gt_box)
        iou_max_idx = torch.argmax(iou)
        iou_max = iou[iou_max_idx]
        if iou_max >= iou_threshold and iou_max_idx not in detected_gt_boxes:
            tp[i] = 1
            detected_gt_boxes.append(iou_max_idx)
    fp = 1 - tp
    return tp, fp


def computeAP(tp, fp, num_gt_class):
    """ Compute Average Precision """
    tp_cumsum = torch.cumsum(tp, dim=0).float()
    fp_cumsum = torch.cumsum(fp, dim=0).float()
    recall = tp_cumsum / (num_gt_class + 1e-16)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
    
    recall = torch.cat([torch.tensor([0.0], device=recall.device), recall, torch.tensor([1.0], device=recall.device)])
    precision = torch.cat([torch.tensor([0.0], device=precision.device), precision, torch.tensor([0.0], device=precision.device)])
    mrec = recall.clone()
    mpre = precision.clone()

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = torch.max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def mAP(predictions, gts, input_shape, ap_each_class, tp_iou_threshold, mode='3D'):
    """ Main function to calculate mAP

    Args:
        predictions: [num_pred, 6 + score + class]
        gts: [num_gt, 6 + class]
    """
    gts = gts[gts[..., :6].any(axis=-1) > 0]
    all_gt_classes = torch.unique(gts[:, 6])
    ap_all = []
    for class_i in all_gt_classes:
        ### NOTE: get the prediction per class and sort it ###
        pred_class = predictions[predictions[..., 7] == class_i]
        pred_class = pred_class[torch.argsort(pred_class[..., 6], descending=True)]
        ### NOTE: get the ground truth per class ###
        gt_class = gts[gts[..., 6] == class_i]
        tp, fp = get_tp_oneclass(pred_class, gt_class, input_shape,
                                 iou_threshold=tp_iou_threshold, mode=mode)
        ap, mrecall, mprecision = computeAP(tp, fp, len(gt_class))
        ap_all.append(ap)
        ap_each_class[int(class_i)].append(ap)
    mean_ap = torch.mean(torch.tensor(ap_all))
    return mean_ap, ap_each_class


def mAP_2d(predictions, gts, input_shape, ap_each_class, tp_iou_threshold):
    """ Main function to calculate mAP

    Args:
        predictions: [num_pred, 4 + score + class]
        gts: [num_gt, 4 + class]
    """
    gts = gts[gts[..., :4].any(axis=-1) > 0]
    all_gt_classes = torch.unique(gts[:, 4])
    ap_all = []
    for class_i in all_gt_classes:
        ### NOTE: get the prediction per class and sort it ###
        pred_class = predictions[predictions[..., 5] == class_i]
        pred_class = pred_class[torch.argsort(pred_class[..., 4], descending=True)]
        ### NOTE: get the ground truth per class ###
        gt_class = gts[gts[..., 4] == class_i]
        tp, fp = get_tp_oneclass(pred_class, gt_class, input_shape,
                                 iou_threshold=tp_iou_threshold, mode="2D")
        ap, mrecall, mprecision = computeAP(tp, fp, len(gt_class))
        ap_all.append(ap)
        ap_each_class[int(class_i)].append(ap)
    mean_ap = torch.mean(torch.tensor(ap_all))
    return mean_ap, ap_each_class