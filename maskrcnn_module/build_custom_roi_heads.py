import torch
import numpy as np
import detectron2

import numpy as np
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from detectron2.structures import Instances
from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from torch.distributions import Normal
import torch.nn.functional as F

class PinballLoss(nn.Module):
    """
    Calculates the quantile loss function.

    Attributes
    ----------
    self.pred : torch.tensor
        Predictions.
    self.target : torch.tensor
        Target to predict.
    self.quantiles : torch.tensor
    """
    def __init__(self, quantile):
        super(PinballLoss, self).__init__()
        self.quantile = quantile

    def forward(self, pred, target):
        """
        Computes the loss for the given prediction.
        """
        error = target - pred
        upper =  self.quantile * error
        lower = (self.quantile - 1) * error

        losses = torch.max(lower, upper)
        # if len(losses.shape) > 1:
        #     loss = torch.mean(torch.sum(losses, dim=1))
        # else:
        loss = torch.mean(losses)
        return loss

class ConvFCHandleHead(nn.Sequential):
    def __init__(
        self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm="", loss_weight=float, predict_std=int, quantile_reg=int
    ):
        
        """
        Converted code from the class FastRCNNConvFCHead, I will basically use the exact same architecture but
        the output will a 3d coordinate for the handle
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        self._loss_weight = loss_weight
        self._predict_std = predict_std
        self._quantile_reg = quantile_reg
        #conv_dims = [256, 256]# todo: optimize this
        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])
        
        #fc_dims = [1024]
        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):      # todo: configure this
            if k == 0:
                self.add_module("flatten", nn.Flatten())
            fc = nn.Linear(int(np.prod(self._output_size)), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.add_module("fc_relu{}".format(k + 1), nn.ReLU())
            self.fcs.append(fc)
            self._output_size = fc_dim

        if self._quantile_reg:
            last_dim = 3*5*3
        elif self._predict_std:
            last_dim = 3*5*2
        else:
            last_dim = 3*5
        fc = nn.Linear(int(np.prod(self._output_size)), last_dim)
        self.add_module("fc{}".format(k + 2), fc)
        self.fcs.append(fc)
        self._output_size = last_dim


        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)




    def forward(self, x, instances: List[Instances]):
        for layer in self:
            x = layer(x)
        
        '''if(self.training):
            gt_handle = [inst.get("gt_handle") for inst in instances if inst.has('gt_handle')]
            gt_handle = torch.cat(gt_handle)
            
            loss_handle = self._loss_weight * torch.mean(torch.norm((gt_handle - x), dim=1))
            return {"loss_handle": loss_handle}
        else: # todo: implement inference
            idx_in_all_det = 0
            #full_handle = torch.cat((x, depths.unsqueeze(1)), dim=1)
            for i in range(len(instances)):
                num_det = instances[i].get("pred_boxes").tensor.shape[0]
                instances[i].set("pred_handle",     x[idx_in_all_det:num_det,:])
                idx_in_all_det += num_det
            return instances
        '''
        num_classes = 5 # number of classes with axis
        if self._quantile_reg:
            x = x.reshape((x.shape[0], num_classes, 3*3))
        elif self._predict_std:
            x = x.reshape((x.shape[0], num_classes, 3*2))
        else:
            x = x.reshape((x.shape[0], num_classes, 3))
        
        if(self.training):
            gt_handle = [inst.get("gt_handle") for inst in instances if inst.has('gt_handle')]
            gt_handle = torch.cat(gt_handle)

            # use the prediction corresponding to the correct class
            gt_class = [inst.get("gt_classes") for inst in instances if inst.has("gt_classes")]
            gt_class = torch.cat(gt_class)

            # predicted handle points
            indices = torch.arange(gt_class.shape[0])
            pred_handle = x[indices, gt_class, :]

            if self._quantile_reg:
                pinball_loss_std_below = PinballLoss(0.1587)
                pinball_loss_median =    PinballLoss(0.5)
                pinball_loss_std_above = PinballLoss(0.8413)
                loss_std_below = pinball_loss_std_below(pred_handle[:, :3], gt_handle)
                loss_median = pinball_loss_median(pred_handle[:, 3:6], gt_handle)
                loss_std_above = pinball_loss_std_above(pred_handle[:, 6:], gt_handle)
                loss_handle = self._loss_weight * ((loss_std_below + loss_median + loss_std_above)/3)
            elif self._predict_std:
                mean_pred = pred_handle[:, :3]
                std_pred = pred_handle[:, 3:]
                std_pred  = F.softplus(std_pred)
                std_pred += 1e-8
                dists = Normal(mean_pred, std_pred)
                loss_handle = -1 * dists.log_prob(gt_handle)
                loss_handle = self._loss_weight * loss_handle.mean()
            else:
                loss_handle = self._loss_weight * torch.mean(torch.norm((gt_handle - pred_handle), dim=1))

            return {"loss_handle": loss_handle}
        else: # todo: implement inference
            idx_in_all_det = 0
            #full_handle = torch.cat((x, depths.unsqueeze(1)), dim=1)
            for i in range(len(instances)):
                num_det = instances[i].get("pred_boxes").tensor.shape[0]
                pred_class = instances[i].get("pred_classes")
                ind = torch.arange(idx_in_all_det, num_det)
                instances[i].set("pred_handle", x[ind, pred_class, :])
                idx_in_all_det += num_det
            return instances
        
class ConvFCSurfnormHead(nn.Sequential):
    def __init__(
        self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm="", loss_weight=float, output_dim=int, predict_std=int, quantile_reg=int
    ):
        
        """
        Converted code from the class FastRCNNConvFCHead, I will basically use the exact same architecture but
        the output will be normalized unit vectors
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        self._loss_weight = loss_weight
        self._predict_std = predict_std
        self._quantile_reg = quantile_reg
        #conv_dims = [256, 256]# todo: optimize this
        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])
        
        #fc_dims = [1024]
        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):      # todo: configure this
            if k == 0:
                self.add_module("flatten", nn.Flatten())
            fc = nn.Linear(int(np.prod(self._output_size)), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.add_module("fc_relu{}".format(k + 1), nn.ReLU())
            self.fcs.append(fc)
            self._output_size = fc_dim

        if self._quantile_reg:
            last_dim = output_dim * 5 * 3 # 3 # 5 # 3
        elif self._predict_std:
            last_dim = output_dim * 5 * 2 # 3 # 5 # 3
        else:
            last_dim = output_dim * 5 # 3 # 5 # 3
        fc = nn.Linear(int(np.prod(self._output_size)), last_dim)
        self.add_module("fc{}".format(k + 2), fc)
        self.fcs.append(fc)
        self._output_size = last_dim


        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)


    def forward(self, x, instances: List[Instances]):
        for layer in self:
            x = layer(x)

        num_classes = 5 # number of classes with axis

        if self._quantile_reg:
            x = x.reshape((x.shape[0], num_classes, 3))
        elif self._predict_std:
            x = x.reshape((x.shape[0], num_classes, 2))
        elif self._output_size == num_classes:
            x = x.reshape((x.shape[0], num_classes, 1))

        if self._output_size == 3:
            # assert False, "not using normals anymore in below code, need to re-implement"
            magnitudes = (torch.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)).unsqueeze(1)
            normals = x[:,:3] / (magnitudes + 1e-12) # add in eps
            #depths = x[:,3]
        
        if(self.training):
            if self._output_size == 3:
                gt_surfnorm = [inst.get("gt_surf_norm") for inst in instances if inst.has('gt_surf_norm')]
            else:
                gt_surfnorm = [inst.get("gt_surf_norm_1d") for inst in instances if inst.has('gt_surf_norm_1d')]
            gt_surfnorm = torch.cat(gt_surfnorm)

            if self._output_size == num_classes:
                gt_surfnorm = gt_surfnorm.unsqueeze(1)

            #loss only contributes for drawers
            gt_class = [inst.get("gt_classes") for inst in instances if inst.has("gt_classes")]
            gt_class = torch.cat(gt_class)

            if self._output_size == 3:
                gt_is_drawer = torch.where(gt_class == 3, 2, 1, 0).type(torch.double) #(gt_class ==3).type(torch.float64)

                loss_surfnorm = torch.matmul((1-torch.nn.CosineSimilarity(dim=1)(gt_surfnorms, normals)).type(torch.double), gt_is_drawer)   # cosine loss

                if torch.sum(gt_is_drawer) > 0:
                    loss_surfnorm = self._loss_weight*loss_surfnorm/torch.sum(gt_is_drawer)
                else:
                    loss_surfnorm *= 0
                #loss_surfnorm = 1 - torch.mean(torch.nn.CosineSimilarity(dim=1)(gt_surfnorms[:,:3], normals))   # cosine loss
                #loss_depth = torch.mean((gt_surfnorms[:,3] - depths)**2)
            else:

                # predicted handle points
                indices = torch.arange(gt_class.shape[0])
                pred_surfnorm = x[indices, gt_class, :]

                if self._quantile_reg:
                    pinball_loss_std_below = PinballLoss(0.1587)
                    pinball_loss_median =    PinballLoss(0.5)
                    pinball_loss_std_above = PinballLoss(0.8413)
                    loss_std_below = pinball_loss_std_below(pred_surfnorm[:, 0], gt_surfnorm)
                    loss_median = pinball_loss_median(pred_surfnorm[:, 1], gt_surfnorm)
                    loss_std_above = pinball_loss_std_above(pred_surfnorm[:, 2], gt_surfnorm)
                    loss_surfnorm = self._loss_weight * ((loss_std_below + loss_median + loss_std_above)/3)
                elif self._predict_std:
                    mean_pred = pred_surfnorm[:, 0].unsqueeze(1)
                    std_pred = pred_surfnorm[:, 1].unsqueeze(1)
                    std_pred  = F.softplus(std_pred)
                    std_pred += 1e-8
                    dists = Normal(mean_pred, std_pred)
                    loss_surfnorm = -1 * dists.log_prob(gt_surfnorm)
                    loss_surfnorm = self._loss_weight * loss_surfnorm.mean()
                else:
                    loss_surfnorm = self._loss_weight * torch.mean(torch.norm((gt_surfnorm - pred_surfnorm), dim=1))

            if not np.isfinite(loss_surfnorm.item()):
                raise FloatingPointError(f"Loss_surfnorm became infinte or NaN")
            return {"loss_surfnorm": loss_surfnorm} #, "loss_depth": loss_depth}
        else: # todo: implement inference
            if self._output_size == 3:
                idx_in_all_det = 0
                #full_surfnorms = torch.cat(normals)
                for i in range(len(instances)):
                    num_det = instances[i].get("pred_boxes").tensor.shape[0]
                    instances[i].set("pred_surf_norm",     normals[idx_in_all_det:num_det,:])
                    idx_in_all_det += num_det
                return instances
            else:
                idx_in_all_det = 0
                #full_handle = torch.cat((x, depths.unsqueeze(1)), dim=1)
                for i in range(len(instances)):
                    num_det = instances[i].get("pred_boxes").tensor.shape[0]
                    pred_class = instances[i].get("pred_classes")
                    ind = torch.arange(idx_in_all_det, num_det)
                    instances[i].set("pred_surf_norm", x[ind, pred_class, :])
                    idx_in_all_det += num_det
                return instances

class ConvFCAxisHead(nn.Sequential):
    def __init__(
        self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm="", loss_weight=float
    ):
        
        """
        Converted code from the class FastRCNNConvFCHead, I will basically use the exact same architecture but
        the output will be two 3d vectors concatenated together
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        self._loss_weight = loss_weight
        #conv_dims = [256, 256]# todo: optimize this
        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])
        # print("FC_DIMS: ", fc_dims)
        # print("LOSS_WEIGHT: ", self._loss_weight)
        #fc_dims = [1024]
        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):      # todo: configure this
            if k == 0:
                self.add_module("flatten", nn.Flatten())
            fc = nn.Linear(int(np.prod(self._output_size)), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.add_module("fc_relu{}".format(k + 1), nn.ReLU())
            self.fcs.append(fc)
            self._output_size = fc_dim


        last_dim = 6*5
        fc = nn.Linear(int(np.prod(self._output_size)), last_dim)
        self.add_module("fc{}".format(k + 2), fc)
        self.fcs.append(fc)
        self._output_size = last_dim


        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x, instances: List[Instances]):
        for layer in self:
            x = layer(x)
        
        num_classes = 5 # number of classes with axis
        
        x = x.reshape((x.shape[0], num_classes, 6))
        
        if(self.training):
            # use the prediction corresponding to the correct class
            gt_class = [inst.get("gt_classes") for inst in instances if inst.has("gt_classes")]
            gt_class = torch.cat(gt_class)

            gt_axis = [inst.get("gt_axis_points") for inst in instances if inst.has('gt_axis_points')]
            gt_axis = torch.cat(gt_axis)
            #ground truth direction of axis
            gt_dir = gt_axis[:,:3] - gt_axis[:,3:]
            magnitudes_gt = (torch.sqrt(gt_dir[:,0]**2 + gt_dir[:,1]**2 + gt_dir[:,2]**2)).unsqueeze(1)
            normals_gt = gt_dir[:,:] / (magnitudes_gt + 1e-12) # add in eps
            
            # predicted axis points
            indices = torch.arange(gt_class.shape[0])
            pred_points = x[indices, gt_class, :]
            directions = pred_points[:, :3] - pred_points[:, 3:]
            magnitudes = (torch.sqrt(directions[:,0]**2 + directions[:,1]**2 + directions[:,2]**2)).unsqueeze(1)
            normals = directions[:,:] / (magnitudes + 1e-12) # add in eps

            # drawers (3) do not have axis points so loss is 0 for drawers
            gt_no_axis = torch.where(gt_class == 3, 0, 1).type(torch.double)
            gt_no_axis = torch.ones_like(gt_no_axis)
            # calculate loss 
            loss_axis_dir = torch.matmul((1-torch.nn.CosineSimilarity(dim=1)(normals_gt, normals)), gt_no_axis)  # cosine loss
            loss_axis_points = torch.matmul(torch.sqrt(torch.sum((gt_axis[:, :3] - pred_points[:, :3])**2, 1)), gt_no_axis)
            loss_axis_points += torch.matmul(torch.sqrt(torch.sum((gt_axis[:, 3:] - pred_points[:, 3:])**2, 1)), gt_no_axis)
        
            loss_axis_dir = self._loss_weight*loss_axis_dir/pred_points.shape[0]
            loss_axis_points = self._loss_weight*loss_axis_points/pred_points.shape[0]
            # if torch.sum(gt_match) > 0:
            #     loss_axis_dir = loss_axis_dir/torch.sum(gt_match)
            #     loss_axis_points = loss_axis_points/torch.sum(gt_match)
            # else:
            #     loss_axis_dir *= 0

            return {"loss_axis_dir": loss_axis_dir, "loss_axis_points": loss_axis_points}
        else: # todo: implement inference
            idx_in_all_det = 0
            #full_axis = torch.cat((x, depths.unsqueeze(1)), dim=1)
            #pred_x = x.reshape((x.shape[0], num_classes, 6))
            for i in range(len(instances)):
                num_det = instances[i].get("pred_boxes").tensor.shape[0]
                pred_class = instances[i].get("pred_classes")
                ind = torch.arange(idx_in_all_det, num_det)
                instances[i].set("pred_axis_points", x[ind, pred_class, :])
                idx_in_all_det += num_det
            return instances


class ConvFCHandleOrientationHead(nn.Sequential):
    def __init__(
        self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm="", loss_weight=float, predict_std=int, quantile_reg=int
    ):
        
        """
        Converted code from the class FastRCNNConvFCHead, I will basically use the exact same architecture but
        the output will a 3d coordinate for the handle
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        self._loss_weight = loss_weight
        self._predict_std = predict_std
        self._quantile_reg = quantile_reg
        #conv_dims = [256, 256]# todo: optimize this
        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])
        
        #fc_dims = [1024]
        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):      # todo: configure this
            if k == 0:
                self.add_module("flatten", nn.Flatten())
            fc = nn.Linear(int(np.prod(self._output_size)), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.add_module("fc_relu{}".format(k + 1), nn.ReLU())
            self.fcs.append(fc)
            self._output_size = fc_dim

        '''
        if self._quantile_reg:
            last_dim = 3*5*3
        elif self._predict_std:
            last_dim = 3*5*2
        else:
            last_dim = 3*5
        '''
        last_dim = 5*4

        fc = nn.Linear(int(np.prod(self._output_size)), last_dim)
        self.add_module("fc{}".format(k + 2), fc)
        self.fcs.append(fc)
        self._output_size = last_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)


    def forward(self, x, instances: List[Instances]):
        for layer in self:
            x = layer(x)
        
        '''if(self.training):
            gt_handle = [inst.get("gt_handle") for inst in instances if inst.has('gt_handle')]
            gt_handle = torch.cat(gt_handle)
            
            loss_handle = self._loss_weight * torch.mean(torch.norm((gt_handle - x), dim=1))
            return {"loss_handle": loss_handle}
        else: # todo: implement inference
            idx_in_all_det = 0
            #full_handle = torch.cat((x, depths.unsqueeze(1)), dim=1)
            for i in range(len(instances)):
                num_det = instances[i].get("pred_boxes").tensor.shape[0]
                instances[i].set("pred_handle",     x[idx_in_all_det:num_det,:])
                idx_in_all_det += num_det
            return instances
        '''
        num_classes = 5 # number of classes with axis
        '''
        if self._quantile_reg:
            x = x.reshape((x.shape[0], num_classes, 3*3))
        elif self._predict_std:
            x = x.reshape((x.shape[0], num_classes, 3*2))
        else:
            x = x.reshape((x.shape[0], num_classes, 3))
        '''
        x = x.reshape((x.shape[0], num_classes, 4))
        
        if(self.training):
            gt_handle_orientation = [inst.get("gt_handle_orientation") for inst in instances if inst.has('gt_handle_orientation')]
            gt_handle_orientation = torch.cat(gt_handle_orientation)

            # use the prediction corresponding to the correct class
            gt_class = [inst.get("gt_classes") for inst in instances if inst.has("gt_classes")]
            gt_class = torch.cat(gt_class)

            # predicted handle points
            indices = torch.arange(gt_class.shape[0])
            pred_handle_orientation = x[indices, gt_class, :]

            '''
            if self._quantile_reg:
                pinball_loss_std_below = PinballLoss(0.1587)
                pinball_loss_median =    PinballLoss(0.5)
                pinball_loss_std_above = PinballLoss(0.8413)
                loss_std_below = pinball_loss_std_below(pred_handle[:, :3], gt_handle)
                loss_median = pinball_loss_median(pred_handle[:, 3:6], gt_handle)
                loss_std_above = pinball_loss_std_above(pred_handle[:, 6:], gt_handle)
                loss_handle = self._loss_weight * ((loss_std_below + loss_median + loss_std_above)/3)
            elif self._predict_std:
                mean_pred = pred_handle[:, :3]
                std_pred = pred_handle[:, 3:]
                std_pred  = F.softplus(std_pred)
                std_pred += 1e-8
                dists = Normal(mean_pred, std_pred)
                loss_handle = -1 * dists.log_prob(gt_handle)
                loss_handle = self._loss_weight * loss_handle.mean()
            else:
                loss_handle = self._loss_weight * torch.mean(torch.norm((gt_handle - pred_handle), dim=1))
            '''
            criterion = nn.CrossEntropyLoss()
            loss_handle_orientation = self._loss_weight * criterion(pred_handle_orientation, gt_handle_orientation)

            return {"loss_handle_orientation": loss_handle_orientation}
        else: # todo: implement inference
            idx_in_all_det = 0
            #full_handle = torch.cat((x, depths.unsqueeze(1)), dim=1)
            for i in range(len(instances)):
                num_det = instances[i].get("pred_boxes").tensor.shape[0]
                pred_class = instances[i].get("pred_classes")
                ind = torch.arange(idx_in_all_det, num_det)
                instances[i].set("pred_handle_orientation", x[ind, pred_class, :])
                idx_in_all_det += num_det
            return instances


def build_surfnorm_head(cfg, input_shape):
    num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
    conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
    num_fc = cfg.NUM_FC #cfg.MODEL.ROI_BOX_HEAD.NUM_FC
    fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
    conv_dims = [conv_dim] * num_conv
    fc_dims = [fc_dim] * num_fc
    conv_norm = cfg.MODEL.ROI_BOX_HEAD.NORM
    loss_weight = cfg.SURFNORM_LOSS
    output_dim = cfg.SURFNORM_OUTPUT_DIM
    predict_std = cfg.PREDICT_STD
    quantile_reg = cfg.QUANTILE_REG
    return ConvFCSurfnormHead(input_shape=input_shape, conv_dims=conv_dims, fc_dims=fc_dims, conv_norm=conv_norm, loss_weight=loss_weight, output_dim=output_dim, predict_std=predict_std, quantile_reg=quantile_reg)

def build_handle_head(cfg, input_shape):
    num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
    conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
    num_fc = cfg.NUM_FC #cfg.MODEL.ROI_BOX_HEAD.NUM_FC
    fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
    conv_dims = [conv_dim] * num_conv
    fc_dims = [fc_dim] * num_fc
    conv_norm = cfg.MODEL.ROI_BOX_HEAD.NORM
    loss_weight = cfg.HANDLE_LOSS
    predict_std = cfg.PREDICT_STD
    quantile_reg = cfg.QUANTILE_REG
    return ConvFCHandleHead(input_shape=input_shape, conv_dims=conv_dims, fc_dims=fc_dims, conv_norm=conv_norm, loss_weight=loss_weight, predict_std=predict_std, quantile_reg=quantile_reg)

def build_axis_head(cfg, input_shape):
    num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
    conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
    num_fc = cfg.NUM_FC #cfg.MODEL.ROI_BOX_HEAD.NUM_FC
    fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
    conv_dims = [conv_dim] * num_conv
    fc_dims = [fc_dim] * num_fc
    conv_norm = cfg.MODEL.ROI_BOX_HEAD.NORM
    loss_weight = cfg.AXIS_LOSS
    return ConvFCAxisHead(input_shape=input_shape, conv_dims=conv_dims, fc_dims=fc_dims, conv_norm=conv_norm, loss_weight=loss_weight)

def build_handle_orientation_head(cfg, input_shape):
    num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
    conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
    num_fc = cfg.NUM_FC #cfg.MODEL.ROI_BOX_HEAD.NUM_FC
    fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
    conv_dims = [conv_dim] * num_conv
    fc_dims = [fc_dim] * num_fc
    conv_norm = cfg.MODEL.ROI_BOX_HEAD.NORM
    loss_weight = cfg.HANDLE_LOSS
    predict_std = cfg.PREDICT_STD
    quantile_reg = cfg.QUANTILE_REG
    return ConvFCHandleOrientationHead(input_shape=input_shape, conv_dims=conv_dims, fc_dims=fc_dims, conv_norm=conv_norm, loss_weight=loss_weight, predict_std=predict_std, quantile_reg=quantile_reg)
