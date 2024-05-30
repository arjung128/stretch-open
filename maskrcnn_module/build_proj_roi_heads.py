import torch
import numpy as np
import detectron2
import math 

import numpy as np
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from detectron2.structures import Instances
from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm

class ConvFCHandleHead(nn.Sequential):
    def __init__(
        self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm="", loss_weight=float
    ):
        
        """
        Converted code from the class FastRCNNConvFCHead, I will basically use the exact same architecture but
        the output will a 3d coordinate for the handle
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
        
        num_classes = 5 # number of classes with axis
        x = x.reshape((x.shape[0], num_classes, 3))
        
        if(self.training):
            gt_handle = [inst.get("gt_handle") for inst in instances if inst.has('gt_handle')]
            gt_handle = torch.cat(gt_handle)
            '''
            hfov = torch.tensor([math.pi/2]).to(torch.device('cuda:0'))
            K = torch.tensor([
                [1 / torch.tan(hfov / 2.).item(), 0., 0., 0.],
                [0., 1 / torch.tan(hfov / 2.).item(), 0., 0.],
                [0., 0.,  1, 0],
                [0., 0., 0, 1]]).to(torch.device('cuda:0'))
            
            # project ground truth handle point
            #print("gt_handle: ", gt_handle.shape)
            gt_handle_expand = torch.cat((gt_handle, torch.ones(gt_handle.shape[0]).unsqueeze(1).to(torch.device('cuda:0'))), dim=1).unsqueeze(2)
            #print("expand: ", gt_handle_expand.shape)
            gt_proj = torch.matmul((K.repeat(gt_handle.shape[0], 1, 1)).type(torch.double), gt_handle_expand).squeeze(2)
            #print("proj: ", gt_proj.shape)
            gt_proj = gt_proj/(-gt_proj[:, 2].unsqueeze(1)-1e-12)
            gt_2d = gt_proj[:, :2]
            #print("2d: ", gt_2d.shape)
            '''
            # use the prediction corresponding to the correct class
            gt_class = [inst.get("gt_classes") for inst in instances if inst.has("gt_classes")]
            gt_class = torch.cat(gt_class)

            # predicted handle points
            indices = torch.arange(gt_class.shape[0])
            pred_handle = x[indices, gt_class, :]

            '''
            pred_handle_expand = torch.cat((pred_handle, torch.ones(pred_handle.shape[0]).unsqueeze(1).to(torch.device('cuda:0'))), dim=1).unsqueeze(2)
            #print("expand: ", gt_handle_expand.shape)
            pred_proj = torch.matmul((K.repeat(pred_handle.shape[0], 1, 1)).type(torch.double), pred_handle_expand.type(torch.double)).squeeze(2)
            #print("proj: ", gt_proj.shape)
            pred_proj = pred_proj/(-pred_proj[:, 2].unsqueeze(1)-1e-12)
            pred_2d = pred_proj[:, :2]
            
            #print("2d: ", pred_2d.shape)
            #gt_class_0 = torch.where(gt_class == 0, 1, 0).type(torch.double)
            proj_dist_loss = torch.norm((gt_2d - pred_2d), dim=1)#*gt_class_0
            thresh = torch.where(proj_dist_loss > 10.0, 0, 1).type(torch.double)
            avg_proj_loss = torch.mean(proj_dist_loss*thresh, dim=0)
            # while proj_dist_loss > 1:
            #     proj_dist_loss = 0.1*proj_dist_loss
            #print(proj_dist_loss)
            '''

            # handle angle loss
            gt_handle_unit = gt_handle/torch.norm(gt_handle)
            pred_handle_unit = pred_handle/torch.norm(pred_handle)
            handle_angle = torch.mean(torch.arccos(torch.sum(gt_handle_unit*pred_handle_unit, axis=1).unsqueeze(1)))
            
            loss_handle = self._loss_weight * (torch.mean(torch.norm((gt_handle - pred_handle), dim=1)) + handle_angle)
            #print("loss: ", loss_handle)
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
        self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm="", loss_weight=float
    ):
        
        """
        Converted code from the class FastRCNNConvFCHead, I will basically use the exact same architecture but
        the output will be normalized unit vectors
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


        last_dim = 3
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
        
        magnitudes = (torch.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)).unsqueeze(1)
        normals = x[:,:3] / (magnitudes + 1e-12) # add in eps
        #depths = x[:,3]
        
        if(self.training):
            gt_surfnorms = [inst.get("gt_surf_norm") for inst in instances if inst.has('gt_surf_norm')]
            gt_surfnorms = torch.cat(gt_surfnorms)
            #loss only contributes for drawers
            gt_class = [inst.get("gt_classes") for inst in instances if inst.has("gt_classes")]
            gt_class = torch.cat(gt_class)
            gt_is_drawer = torch.where(gt_class == 3, 1, 0).type(torch.double) #(gt_class ==3).type(torch.float64)

            loss_surfnorm = torch.matmul((1-torch.nn.CosineSimilarity(dim=1)(gt_surfnorms, normals)).type(torch.double), gt_is_drawer)   # cosine loss
            if torch.sum(gt_is_drawer) > 0:
                loss_surfnorm = self._loss_weight*loss_surfnorm/torch.sum(gt_is_drawer)
            else:
                loss_surfnorm *= 0
            #loss_surfnorm = 1 - torch.mean(torch.nn.CosineSimilarity(dim=1)(gt_surfnorms[:,:3], normals))   # cosine loss
            #loss_depth = torch.mean((gt_surfnorms[:,3] - depths)**2)
            if not np.isfinite(loss_surfnorm.item()):
                raise FloatingPointError(f"Loss_surfnorm became infinte or NaN")
            return {"loss_surfnorm": loss_surfnorm} #, "loss_depth": loss_depth}
        else: # todo: implement inference
            idx_in_all_det = 0
            #full_surfnorms = torch.cat(normals)
            for i in range(len(instances)):
                num_det = instances[i].get("pred_boxes").tensor.shape[0]
                instances[i].set("pred_surf_norm",     normals[idx_in_all_det:num_det,:])
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

'''
class ConvFCAxis0Head(nn.Sequential):
    def __init__(
        self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm=""
    ):
        
        """
        Converted code from the class FastRCNNConvFCHead, I will basically use the exact same architecture but
        the output will be two 3d vectors concatenated together
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        
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


        last_dim = 6
        fc = nn.Linear(int(np.prod(self._output_size)), last_dim)
        self.add_module("fc{}".format(k + 2), fc)
        self.fcs.append(fc)
        self._output_size = last_dim


        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x, instances: List[Instances]):
        loss_weight = 0.00001 # hyperparam
        for layer in self:
            x = layer(x)
        
        directions = x[:, :3] - x[:, 3:]
        magnitudes = (torch.sqrt(directions[:,0]**2 + directions[:,1]**2 + directions[:,2]**2)).unsqueeze(1)
        normals = directions[:,:] / (magnitudes + 1e-12) # add in eps
        
        if(self.training):
            gt_axis = [inst.get("gt_axis_points") for inst in instances if inst.has('gt_axis_points')]
            gt_axis = torch.cat(gt_axis)
            #ground truth direction of axis
            gt_dir = gt_axis[:,:3] - gt_axis[:,3:]
            magnitudes_gt = (torch.sqrt(gt_dir[:,0]**2 + gt_dir[:,1]**2 + gt_dir[:,2]**2)).unsqueeze(1)
            normals_gt = gt_dir[:,:] / (magnitudes_gt + 1e-12) # add in eps
            # loss only contributes for non drawers
            gt_class = [inst.get("gt_classes") for inst in instances if inst.has("gt_classes")]
            gt_class = torch.cat(gt_class)
            gt_match = (gt_class==0).type(torch.double)
            # calculate loss 
            loss_axis_dir = torch.matmul((1-torch.nn.CosineSimilarity(dim=1)(normals_gt, normals)), gt_match)  # cosine loss
            loss_axis_points = torch.matmul(torch.sqrt(torch.sum((gt_axis[:, :3] - x[:, :3])**2, 1)), gt_match)
            loss_axis_points += torch.matmul(torch.sqrt(torch.sum((gt_axis[:, 3:] - x[:, 3:])**2, 1)), gt_match)
        
            if torch.sum(gt_match) > 0:
                loss_axis_dir = loss_axis_dir/torch.sum(gt_match)
                loss_axis_points = loss_axis_points/torch.sum(gt_match)
            else:
                loss_axis_dir *= 0

            return {"loss_axis0_dir": loss_axis_dir, "loss_axis0_points": loss_axis_points}
        else: # todo: implement inference
            idx_in_all_det = 0
            #full_axis = torch.cat((x, depths.unsqueeze(1)), dim=1)
            for i in range(len(instances)):
                num_det = instances[i].get("pred_boxes").tensor.shape[0]
                instances[i].set("pred_axis0_points",     x[idx_in_all_det:num_det,:])
                idx_in_all_det += num_det
            return instances

class ConvFCAxis1Head(nn.Sequential):
    def __init__(
        self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm=""
    ):
        
        """
        Converted code from the class FastRCNNConvFCHead, I will basically use the exact same architecture but
        the output will be two 3d vectors concatenated together
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        
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


        last_dim = 6
        fc = nn.Linear(int(np.prod(self._output_size)), last_dim)
        self.add_module("fc{}".format(k + 2), fc)
        self.fcs.append(fc)
        self._output_size = last_dim


        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x, instances: List[Instances]):
        loss_weight = 0.00001 # hyperparam
        for layer in self:
            x = layer(x)
        
        directions = x[:, :3] - x[:, 3:]
        magnitudes = (torch.sqrt(directions[:,0]**2 + directions[:,1]**2 + directions[:,2]**2)).unsqueeze(1)
        normals = directions[:,:] / (magnitudes + 1e-12) # add in eps
        
        if(self.training):
            gt_axis = [inst.get("gt_axis_points") for inst in instances if inst.has('gt_axis_points')]
            gt_axis = torch.cat(gt_axis)
            #ground truth direction of axis
            gt_dir = gt_axis[:,:3] - gt_axis[:,3:]
            magnitudes_gt = (torch.sqrt(gt_dir[:,0]**2 + gt_dir[:,1]**2 + gt_dir[:,2]**2)).unsqueeze(1)
            normals_gt = gt_dir[:,:] / (magnitudes_gt + 1e-12) # add in eps
            # loss only contributes for non drawers
            gt_class = [inst.get("gt_classes") for inst in instances if inst.has("gt_classes")]
            gt_class = torch.cat(gt_class)
            gt_match = (gt_class == 1).type(torch.double)
            # calculate loss 
            loss_axis_dir = torch.matmul((1-torch.nn.CosineSimilarity(dim=1)(normals_gt, normals)), gt_match)  # cosine loss
            loss_axis_points = torch.matmul(torch.sqrt(torch.sum((gt_axis[:, :3] - x[:, :3])**2, 1)), gt_match)
            loss_axis_points += torch.matmul(torch.sqrt(torch.sum((gt_axis[:, 3:] - x[:, 3:])**2, 1)), gt_match)
        
            if torch.sum(gt_match) > 0:
                loss_axis_dir = loss_axis_dir/torch.sum(gt_match)
                loss_axis_points = loss_axis_points/torch.sum(gt_match)
            else:
                loss_axis_dir *= 0

            return {"loss_axis1_dir": loss_axis_dir, "loss_axis1_points": loss_axis_points}
        else: # todo: implement inference
            idx_in_all_det = 0
            #full_axis = torch.cat((x, depths.unsqueeze(1)), dim=1)
            for i in range(len(instances)):
                num_det = instances[i].get("pred_boxes").tensor.shape[0]
                instances[i].set("pred_axis1_points",     x[idx_in_all_det:num_det,:])
                idx_in_all_det += num_det
            return instances

class ConvFCAxis2Head(nn.Sequential):
    def __init__(
        self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm=""
    ):
        
        """
        Converted code from the class FastRCNNConvFCHead, I will basically use the exact same architecture but
        the output will be two 3d vectors concatenated together
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        
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


        last_dim = 6
        fc = nn.Linear(int(np.prod(self._output_size)), last_dim)
        self.add_module("fc{}".format(k + 2), fc)
        self.fcs.append(fc)
        self._output_size = last_dim


        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x, instances: List[Instances]):
        loss_weight = 0.00001 # hyperparam
        for layer in self:
            x = layer(x)
        
        directions = x[:, :3] - x[:, 3:]
        magnitudes = (torch.sqrt(directions[:,0]**2 + directions[:,1]**2 + directions[:,2]**2)).unsqueeze(1)
        normals = directions[:,:] / (magnitudes + 1e-12) # add in eps
        
        if(self.training):
            gt_axis = [inst.get("gt_axis_points") for inst in instances if inst.has('gt_axis_points')]
            gt_axis = torch.cat(gt_axis)
            #ground truth direction of axis
            gt_dir = gt_axis[:,:3] - gt_axis[:,3:]
            magnitudes_gt = (torch.sqrt(gt_dir[:,0]**2 + gt_dir[:,1]**2 + gt_dir[:,2]**2)).unsqueeze(1)
            normals_gt = gt_dir[:,:] / (magnitudes_gt + 1e-12) # add in eps
            # loss only contributes for non drawers
            gt_class = [inst.get("gt_classes") for inst in instances if inst.has("gt_classes")]
            gt_class = torch.cat(gt_class)
            gt_match = (gt_class == 2).type(torch.double)
            # calculate loss 
            loss_axis_dir = torch.matmul((1-torch.nn.CosineSimilarity(dim=1)(normals_gt, normals)), gt_match)  # cosine loss
            loss_axis_points = torch.matmul(torch.sqrt(torch.sum((gt_axis[:, :3] - x[:, :3])**2, 1)), gt_match)
            loss_axis_points += torch.matmul(torch.sqrt(torch.sum((gt_axis[:, 3:] - x[:, 3:])**2, 1)), gt_match)
        
            if torch.sum(gt_match) > 0:
                loss_axis_dir = loss_axis_dir/torch.sum(gt_match)
                loss_axis_points = loss_axis_points/torch.sum(gt_match)
            else:
                loss_axis_dir *= 0

            return {"loss_axis2_dir": loss_axis_dir, "loss_axis2_points": loss_axis_points}
        else: # todo: implement inference
            idx_in_all_det = 0
            #full_axis = torch.cat((x, depths.unsqueeze(1)), dim=1)
            for i in range(len(instances)):
                num_det = instances[i].get("pred_boxes").tensor.shape[0]
                instances[i].set("pred_axis2_points",     x[idx_in_all_det:num_det,:])
                idx_in_all_det += num_det
            return instances

class ConvFCAxis4Head(nn.Sequential):
    def __init__(
        self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm=""
    ):
        
        """
        Converted code from the class FastRCNNConvFCHead, I will basically use the exact same architecture but
        the output will be two 3d vectors concatenated together
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        
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


        last_dim = 6
        fc = nn.Linear(int(np.prod(self._output_size)), last_dim)
        self.add_module("fc{}".format(k + 2), fc)
        self.fcs.append(fc)
        self._output_size = last_dim


        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x, instances: List[Instances]):
        loss_weight = 0.00001 # hyperparam
        for layer in self:
            x = layer(x)
        
        directions = x[:, :3] - x[:, 3:]
        magnitudes = (torch.sqrt(directions[:,0]**2 + directions[:,1]**2 + directions[:,2]**2)).unsqueeze(1)
        normals = directions[:,:] / (magnitudes + 1e-12) # add in eps
        
        if(self.training):
            gt_axis = [inst.get("gt_axis_points") for inst in instances if inst.has('gt_axis_points')]
            gt_axis = torch.cat(gt_axis)
            #ground truth direction of axis
            gt_dir = gt_axis[:,:3] - gt_axis[:,3:]
            magnitudes_gt = (torch.sqrt(gt_dir[:,0]**2 + gt_dir[:,1]**2 + gt_dir[:,2]**2)).unsqueeze(1)
            normals_gt = gt_dir[:,:] / (magnitudes_gt + 1e-12) # add in eps
            # loss only contributes for non drawers
            gt_class = [inst.get("gt_classes") for inst in instances if inst.has("gt_classes")]
            gt_class = torch.cat(gt_class)
            gt_match = (gt_class==4).type(torch.double)
            # calculate loss 
            loss_axis_dir = torch.matmul((1-torch.nn.CosineSimilarity(dim=1)(normals_gt, normals)), gt_match)  # cosine loss
            loss_axis_points = torch.matmul(torch.sqrt(torch.sum((gt_axis[:, :3] - x[:, :3])**2, 1)), gt_match)
            loss_axis_points += torch.matmul(torch.sqrt(torch.sum((gt_axis[:, 3:] - x[:, 3:])**2, 1)), gt_match)
        
            if torch.sum(gt_match) > 0:
                loss_axis_dir = loss_axis_dir/torch.sum(gt_match)
                loss_axis_points = loss_axis_points/torch.sum(gt_match)
            else:
                loss_axis_dir *= 0

            return {"loss_axis4_dir": loss_axis_dir, "loss_axis4_points": loss_axis_points}
        else: # todo: implement inference
            idx_in_all_det = 0
            #full_axis = torch.cat((x, depths.unsqueeze(1)), dim=1)
            for i in range(len(instances)):
                num_det = instances[i].get("pred_boxes").tensor.shape[0]
                instances[i].set("pred_axis4_points",     x[idx_in_all_det:num_det,:])
                idx_in_all_det += num_det
            return instances
'''


def build_surfnorm_head(cfg, input_shape):
    num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
    conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
    num_fc = cfg.NUM_FC #cfg.MODEL.ROI_BOX_HEAD.NUM_FC
    fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
    conv_dims = [conv_dim] * num_conv
    fc_dims = [fc_dim] * num_fc
    conv_norm = cfg.MODEL.ROI_BOX_HEAD.NORM
    loss_weight = cfg.SURFNORM_LOSS
    return ConvFCSurfnormHead(input_shape=input_shape, conv_dims=conv_dims, fc_dims=fc_dims, conv_norm=conv_norm, loss_weight=loss_weight)

def build_handle_head(cfg, input_shape):
    num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
    conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
    num_fc = cfg.NUM_FC #cfg.MODEL.ROI_BOX_HEAD.NUM_FC
    fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
    conv_dims = [conv_dim] * num_conv
    fc_dims = [fc_dim] * num_fc
    conv_norm = cfg.MODEL.ROI_BOX_HEAD.NORM
    loss_weight = cfg.HANDLE_LOSS
    return ConvFCHandleHead(input_shape=input_shape, conv_dims=conv_dims, fc_dims=fc_dims, conv_norm=conv_norm, loss_weight=loss_weight)

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

'''
def build_axis0_head(cfg, input_shape):
    num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
    conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
    num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
    fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
    conv_dims = [conv_dim] * num_conv
    fc_dims = [fc_dim] * num_fc
    conv_norm = cfg.MODEL.ROI_BOX_HEAD.NORM
    
    return ConvFCAxis0Head(input_shape=input_shape, conv_dims=conv_dims, fc_dims=fc_dims, conv_norm=conv_norm)

def build_axis1_head(cfg, input_shape):
    num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
    conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
    num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
    fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
    conv_dims = [conv_dim] * num_conv
    fc_dims = [fc_dim] * num_fc
    conv_norm = cfg.MODEL.ROI_BOX_HEAD.NORM
    
    return ConvFCAxis1Head(input_shape=input_shape, conv_dims=conv_dims, fc_dims=fc_dims, conv_norm=conv_norm)

def build_axis2_head(cfg, input_shape):
    num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
    conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
    num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
    fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
    conv_dims = [conv_dim] * num_conv
    fc_dims = [fc_dim] * num_fc
    conv_norm = cfg.MODEL.ROI_BOX_HEAD.NORM
    
    return ConvFCAxis2Head(input_shape=input_shape, conv_dims=conv_dims, fc_dims=fc_dims, conv_norm=conv_norm)

def build_axis4_head(cfg, input_shape):
    num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
    conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
    num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
    fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
    conv_dims = [conv_dim] * num_conv
    fc_dims = [fc_dim] * num_fc
    conv_norm = cfg.MODEL.ROI_BOX_HEAD.NORM
    
    return ConvFCAxis4Head(input_shape=input_shape, conv_dims=conv_dims, fc_dims=fc_dims, conv_norm=conv_norm)

'''