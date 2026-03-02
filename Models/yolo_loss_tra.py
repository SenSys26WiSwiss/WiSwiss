import torch
import torch.nn as nn
from .raddet_utils import iou2d

def extract_yolo_info_2d(yolo_format_data):
    """ Extract box, objectness, class from yolo format data """
    box = yolo_format_data[..., :4]
    conf = yolo_format_data[..., 4:5]
    category = yolo_format_data[..., 5:]
    return box, conf, category

class RADDetLoss_2d(nn.Module):
    def __init__(self, input_shape, focal_loss_iou_threshold):
        super(RADDetLoss_2d, self).__init__()
        self.input_shape = input_shape
        self.focal_loss_iou_threshold = focal_loss_iou_threshold
        self.bce_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    
    def forward(self, pred_raw, pred, label, raw_gt_boxes):
        assert len(raw_gt_boxes.shape) == 3
        assert pred_raw.shape == label.shape
        assert pred_raw.shape[0] == len(raw_gt_boxes)
        assert pred.shape == label.shape
        assert pred.shape[0] == len(raw_gt_boxes)

        raw_box, raw_conf, raw_category = extract_yolo_info_2d(pred_raw)
        pred_box, pred_conf, pred_category = extract_yolo_info_2d(pred)
        gt_box, gt_conf, gt_category = extract_yolo_info_2d(label)
        
        iou_loss = self.yolo1_loss(pred_box, gt_box, gt_conf, self.input_shape, False)
        focal_loss = self.focal_loss(raw_conf, pred_conf, gt_conf, pred_box, raw_gt_boxes, 
                                     self.focal_loss_iou_threshold)
        category_loss = self.category_loss(raw_category, gt_category, gt_conf)

        total_iou_loss = torch.mean(torch.sum(iou_loss, dim=[1, 2, 3]))
        total_focal_loss = torch.mean(torch.sum(focal_loss, dim=[1, 2, 3]))
        total_category_loss = torch.mean(torch.sum(category_loss, dim=[1, 2, 3]))
        
        return total_iou_loss, total_focal_loss, total_category_loss
    
    def yolo1_loss(self, pred_box, gt_box, gt_conf, input_shape, box_loss_scale=False):
        """ loss function for box regression \cite{YOLOV1} """
        assert pred_box.shape == gt_box.shape
        if box_loss_scale:
            box_area = gt_box[...,2] * gt_box[...,3]
            input_area = input_shape[1] * input_shape[2]
            scale = 2.0 - 1.0 * box_area / input_area
        else:
            scale = 1.0
        ### NOTE: YOLOv1 original loss function ###
        iou_loss = gt_conf*scale * (torch.square(pred_box[..., :2] - gt_box[..., :2]) + 
                                    torch.square(torch.sqrt(pred_box[..., 2:]) - torch.sqrt(gt_box[..., 2:])))
    
        return iou_loss
    
    def focal_loss(self, raw_conf, pred_conf, gt_conf, pred_box, raw_gt_boxes, iou_loss_threshold=0.5):
        """
        Calculate focal loss for objectness
        shape of iou, max_iou, gt_conf_negative, conf_focal, focal_loss
        [3, 16, 16, num_anchors, 30], [3, 16, 16, num_anchors, 1], [3, 16, 16, num_anchors, 1], [3, 16, 16, num_anchors, 1], [3, 16, 16, num_anchors, 1]
        """
        iou = iou2d(torch.unsqueeze(pred_box, dim=-2), raw_gt_boxes[:, None, None, None, :, :])
        max_iou = torch.unsqueeze(torch.max(iou, dim=-1)[0], dim=-1)

        gt_conf_negative = (1.0 - gt_conf) * (max_iou < iou_loss_threshold).to(torch.float32)
        conf_focal = torch.pow(gt_conf - pred_conf, 2.0)
        alpha = 0.01
        focal_loss = conf_focal * (gt_conf * self.bce_criterion(input=raw_conf, target=gt_conf) + 
                                   alpha * gt_conf_negative * self.bce_criterion(input=raw_conf, target=gt_conf))
        return focal_loss


    def category_loss(self, raw_category, gt_category, gt_conf):
        """ loss function for category classification """
        category_loss = gt_conf * self.bce_criterion(input=raw_category, target=gt_category)
        return category_loss