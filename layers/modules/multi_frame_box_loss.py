# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..box_utils import match, log_sum_exp

class MultiFrameBoxLoss(nn.Module):
    def __init__(self):
        pass

    def forward(self, predictions, targets):
        """
        Args:
            predictions: a tuple containing loc, conf and
            anchor boxes.
                conf.shape: torch.size(batch_size, num_frames, num_anchors, 2)
                loc.shpae: torch.size(batch_size, num_frames, num_anchors, 4)
                anchor.shape: torch.size(num_anchors, 4)

            targets: groundtruth boxes and labels for a batch.
                targets.shape: [batch_size, num_frames, num_objs, 5]
                (5 = loc(4) + label(1))
        """
        loc_data, conf_data, anchors = predictions
        batch_size, num_frames, num_anchors = loc_data.size()[:, -2]

        # match anchor boxes with groundtruth boxes
        loc_t = torch.Tensor(batch_size, num_frames, num_anchors, 4)
        conf_t = torch.Tensor(batch_size, num_frames, num_anchors)
        for batch in range(batch_size):
            for frame in range(num_frames):
                truths = targets[batch][frame][:, :-1].data
                labels = targets[batch][frame][:, -1].data
                defaults = anchors.data
                match_frame(
                    self.threshold,
                    truths,
                    defaults,
                    self.variances,
                    labels,
                    loc_t,
                    conf_t,
                    batch,
                    frame
                )