# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..box_utils import match, log_sum_exp

class MultiFrameBoxLoss(nn.Module):
    def __init__(self, np_ratio, overlap_threshold, variance, num_frames, num_classes):
        super(MultiFrameBoxLoss, self).__init__()
        
        self.threshold = overlap_threshold
        self.variance = variance
        self.neg_pos_ratio = np_ratio
        self.num_frames = num_frames
        self.num_classes = num_classes

    def forward(self, predictions, targets):
        """
        Args:
            predictions: a tuple containing loc, conf and
            anchor boxes.
                conf.shape: torch.size(batch_size, num_anchors * num_frames, num_classes)
                loc.shape: torch.size(batch_size, num_anchors * num_frames, 4)
                anchor.shape: torch.size(num_anchors, 4)

            targets: groundtruth boxes and labels for a batch.
                targets.shape: [batch_size, num_frames, num_objs, 5]
                (5 = loc(4) + label(1))
        """
        loc_data, conf_data, anchors = predictions
        size = loc_data.size()
        batch_size = size[0]
        num_data = size[1]
        num_anchors = num_data // self.num_frames

        # match anchor boxes with groundtruth boxes
        loc_t = torch.Tensor(batch_size * self.num_frames, num_anchors, 4)
        conf_t = torch.Tensor(batch_size * self.num_frames, num_anchors)

        for batch in range(batch_size):
            for frame in range(self.num_frames):
                frame_targets = targets[batch][frame]
                truths = frame_targets[:, :-1]
                labels = frame_targets[:, -1]

                defaults = anchors # .detach()

                match(
                    self.threshold,
                    truths,
                    defaults,
                    self.variance,
                    labels,
                    loc_t,
                    conf_t,
                    batch * self.num_frames + frame
                )
                
        conf_mask = conf_t > 0

        # Smooth L1 - Lreg
        loc_mask = conf_mask.view(batch_size, -1).unsqueeze(conf_mask.dim()).expand_as(loc_data)
        
        loc_pred = loc_data[loc_mask].view(-1, 4)
        loc_t = loc_t.view(batch_size, -1, 4)[loc_mask].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_pred, loc_t, size_average=False)

        # hard negative mining (only for Lcls)
        flat_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(flat_conf) - flat_conf.gather(1, conf_t.view(-1, 1))

        # we should consider frame because predictions are always harder than detection
        # and we don't want only pick prediction losses
        # shape: [batch_size, num_frames, num_anchors]
        loss_c = loss_c.squeeze().view(-1, self.num_frames, num_anchors)
        conf_mask = conf_mask.view(-1, self.num_frames, num_anchors)

        # filter out posivite sample so they won't be sorted
        loss_c[conf_mask] = 0

        # get rank of each item in loss_c
        _, loss_idx = loss_c.sort(2, descending=True)
        _, idx_rank = loss_idx.sort(2)

        # calc hard negative count in each frame
        # shape: [batch_size, num_frames, 1]
        num_pos = conf_mask.long().sum(2, keepdim=True)
        num_neg = torch.clamp(
            self.neg_pos_ratio * num_pos,
            max=conf_mask.size(2) - 1
        )

        # ok, now get negative conf mask
        # shape: [batch_size, num_frames, num_anchors]
        neg_mask = idx_rank < num_neg.expand_as(idx_rank)

        # masks for network output
        # shape: [batch_size, num_frames * num_anchors, 2]
        pos_idx = conf_mask.view(-1, self.num_frames * num_anchors).unsqueeze(2).expand_as(conf_data)
        neg_idx = neg_mask.view(-1, self.num_frames * num_anchors).unsqueeze(2).expand_as(conf_data)

        # select positive and hard negative samples from network output and target
        # shape: [batch_size, num_frames * num_anchors]
        conf_pred = conf_data[(pos_idx + neg_idx).gt(0)].argmax(dim=conf_data.dim() - 1, keepdim=False)

        selected_targets= conf_t.view(batch_size, -1)[(
            conf_mask.view(batch_size, -1) +
            neg_mask.view(batch_size, -1)
        ).gt(0)]

        # Lcls
        loss_c = F.binary_cross_entropy(
            conf_pred,
            selected_targets,
            size_average=False
        )

        return loss_l, loss_c
