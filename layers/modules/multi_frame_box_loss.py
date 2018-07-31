# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..box_utils import match, log_sum_exp

class MultiFrameBoxLoss(nn.Module):
    def __init__(self, np_ratio, overlap_threshold, variance):
        self.threshold = overlap_threshold
        self.variance = variance
        self.neg_pos_ratio = np_ratio

    def forward(self, predictions, targets):
        """
        Args:
            predictions: a tuple containing loc, conf and
            anchor boxes.
                conf.shape: torch.size(batch_size, num_anchors * num_frames, 2)
                loc.shpae: torch.size(batch_size, num_anchors * num_frames, 4)
                anchor.shape: torch.size(num_anchors, 4)

            targets: groundtruth boxes and labels for a batch.
                targets.shape: [batch_size, num_frames, num_objs, 5]
                (5 = loc(4) + label(1))
        """
        loc_data, conf_data, anchors = predictions
        batch_size, num_frames, num_anchors = loc_data.size()[:, -2]

        # match anchor boxes with groundtruth boxes
        loc_t = torch.Tensor(batch_size * num_frames, num_anchors, 4)
        conf_t = torch.Tensor(batch_size * num_frames, num_anchors)

        for batch in range(batch_size):
            for frame in range(num_frames):
                truths = targets[batch][frame][:, :-1].data
                labels = targets[batch][frame][:, -1].data
                defaults = anchors.data

                match(
                    self.threshold,
                    truths,
                    defaults,
                    self.variances,
                    labels,
                    loc_t,
                    conf_t,
                    batch * num_frames + frame
                )

        conf_mask = conf_t > 0

        # Smooth L1 - Lreg
        loc_mask = conf_mask.unsqueeze(conf_mask.dim()).expand_as(loc_data)
        loc_pred = loc_data[loc_mask].view(-1, 4)
        loc_t = loc_t[loc_mask].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_pred, loc_t, size_average=False)

        # hard negative mining (only for Lcls)
        flat_conf = conf_data.view(-1, 2)
        loss_c = log_sum_exp(flat_conf) - flat_conf.gather(1, conf_t.view(-1, 1))

        loss_c[conf_mask] = 0 # filter out posivite sample so they won't be sorted

        # we should consider frame because predictions are always harder than detection
        # and we don't want only pick prediction losses
        loss_c = loss_c.view(batch_size, num_frames, -1)

        # get rank of each item in loss_c
        _, loss_idx = loss_c.sort(2, descending=True)
        _, idx_rank = loss_idx.sort(2)

        # calc hard negative count in each frame
        num_pos = conf_mask.long().sum(2, keepdim=True)
        num_neg = torch.clamp(self.neg_pos_ratio * num_pos, max=conf_mask.size(2) - 1)

        # ok, now get negative conf mask
        neg_mask = idx_rank < num_neg.expand_as(idx_rank)

        # masks for network output
        pos_idx = conf_mask.unsqueeze(3).expand_as(conf_data)
        neg_idx = neg_mask.unsqueeze(3).expand_as(conf_data)

        # select positive and hard negative samples from network output and target
        conf_pred = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, 2)
        selected_targets= conf_t[(conf_mask + neg_mask).gt(0)]

        # Lcls
        loss_c = F.cross_entropy(conf_pred, selected_targets, size_average=False)

        return loss_l, loss_c

