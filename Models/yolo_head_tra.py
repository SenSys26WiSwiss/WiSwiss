import torch
import torch.nn as nn
from einops import rearrange
from .raddet_utils import iou2d


class detection2d_head(nn.Module):
    def __init__(self, num_anchors, num_classes, grid_shape):
        super(detection2d_head, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.grid_shape = grid_shape
        
        final_output_channels = int(num_anchors * (num_classes + 5))
        self.final_shape = [-1, final_output_channels, grid_shape[1], grid_shape[2]]
        
        self.conv1 = nn.Conv2d(in_channels=grid_shape[0], 
                            out_channels=grid_shape[0]*2, 
                            kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(grid_shape[0]*2)
        
        self.conv2 = nn.Conv2d(in_channels=grid_shape[0]*2, 
                               out_channels=final_output_channels, 
                               kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, input_feature):
        x = self.relu(self.bn1(self.conv1(input_feature)))
        x = self.conv2(x)

        return x


def decode_yolo_2d(yolo_output, grid_strides, anchors, scale, num_classes=6):
    x = rearrange(yolo_output, 'n c h w -> n h w c')
    anchors = torch.tensor(anchors, device=yolo_output.device)
    grid_strides = torch.tensor(grid_strides, device=yolo_output.device)
    grid_size = x.shape[1:3]
    bs = yolo_output.shape[0]
    
    g0, g1 = grid_size
    pred_raw = rearrange(x, 'n h w (c c1) -> n h w c c1', c1=num_classes+5)
    raw_xy = pred_raw[:, :, :, :, :2]
    raw_wh = pred_raw[:, :, :, :, 2:4]
    raw_conf = pred_raw[:, :, :, :, 4:5]
    raw_prob = pred_raw[:, :, :, :, 5:]
    
    xx, yy = torch.meshgrid(torch.arange(g0), torch.arange(g1), indexing='ij')
    xy_grid = torch.stack([xx, yy], dim=-1).to(yolo_output.device)
    xy_grid = torch.unsqueeze(xy_grid, 2)
    xy_grid = torch.unsqueeze(xy_grid, 0)
    xy_grid = torch.tile(xy_grid, (bs, 1, 1, len(anchors), 1)).to(torch.float32)

    # This is a scaling trick (seen in YOLOv4) to allow the model to predict centers slightly outside the grid cell 
    # (improving detection of objects near cell boundaries).
    # If scale = 1, the formula reduces to sigmoid(raw_xyz), strictly keeping the center in [0, 1].
    # If scale > 1, the center can shift beyond the cell (e.g., scale=2 allows the center to move up to ±0.5 cells away).
    pred_xy = ((torch.sigmoid(raw_xy) * scale) - 0.5 * (scale - 1) + xy_grid) * grid_strides[1:]
    raw_wh = torch.clamp(raw_wh, 1e-12, 1e12)
    pred_wh = torch.exp(raw_wh) * anchors
    pred_xywh = torch.cat((pred_xy, pred_wh), dim=-1)

    pred_conf = torch.sigmoid(raw_conf)
    pred_prob = torch.sigmoid(raw_prob)
    
    results = torch.cat((pred_xywh, pred_conf, pred_prob), dim=-1)
    return pred_raw, results


def yol2predictions_2d(yolo_output, conf_threshold=0.5):
    """ Transfer YOLO output to [:, 6], where 6 means
    [x, y, w, h, score, class_index]"""
    prediction = yolo_output.reshape(-1, yolo_output.shape[-1])
    prediction_class = torch.argmax(prediction[:, 5:], dim=-1, keepdim=True)
    # Concatenate coordinates, confidence, and class indices
    predictions = torch.cat([prediction[:, :5], prediction_class.float()], dim=-1)
    # Apply confidence threshold mask
    conf_mask = (predictions[:, 4] >= conf_threshold)
    predictions = predictions[conf_mask]
    return predictions


def nms_2d(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """ Bboxes format [x, y, w, h, score, class_index] """
    """ Implemented the same way as YOLOv4 """
    assert method in ['nms', 'soft-nms']
    device = bboxes.device
    
    if len(bboxes) == 0:
        best_bboxes = torch.zeros((0, 6), device=device)
    else:
        all_pred_classes = torch.unique(bboxes[:, 5])
        best_bboxes = []
        for cls in all_pred_classes:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]

            while len(cls_bboxes) > 0:
                max_ind = torch.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                # Remove the selected box from consideration
                cls_bboxes = torch.cat([cls_bboxes[:max_ind], cls_bboxes[max_ind+1:]])
                # Calculate IoU with remaining boxes
                iou = iou2d(best_bbox[:4].unsqueeze(0), cls_bboxes[:, :4])
                weight = torch.ones(len(iou), device=device)
                
                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0
                elif method == 'soft-nms':
                    weight = torch.exp(-(1.0 * iou ** 2 / sigma))
                # Update scores
                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                # Filter out boxes with score <= 0
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]
        
        if len(best_bboxes) > 0:
            best_bboxes = torch.stack(best_bboxes)
        else:
            best_bboxes = torch.zeros((0, 6), device=device)
    return best_bboxes
