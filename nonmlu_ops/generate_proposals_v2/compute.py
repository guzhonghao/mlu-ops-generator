import numpy as np
import paddle.fluid as fluid
import paddle
import sys
import random
from nonmlu_ops.base import *

@registerTensorList('generate_proposals_v2')
class PolyNmsTensorList(TensorList):
    pass

@registerOp('generate_proposals_v2')
class PolyNmsOp(OpTest):
    def __init__(self, tensor_list, params):
        super().__init__(tensor_list, params)
        self.pre_nms_top_n = self.params_.get("pre_nms_top_n", 6000)
        self.post_nms_top_n = self.params_.get("post_nms_top_n", 2000)
        self.nms_thresh = self.params_.get("nms_thresh", 0.5)
        self.min_size = self.params_.get("min_size", 0.1)
        self.eta = self.params_.get("eta", 1.0)
        self.pixel_offset = self.params_.get("pixel_offset", True)
        print("__init generate_proposals_v2")

    def compute(self):
        paddle.disable_static()

        dtype = self.tensor_list_.getInputTensor(0).getDataType()
        np_scores = self.tensor_list_.getInputTensor(0).getData()
        np_bboox_deltas = self.tensor_list_.getInputTensor(1).getData()
        np_anchors = self.tensor_list_.getInputTensor(2).getData()
        np_variances = self.tensor_list_.getInputTensor(3).getData()
        np_img_size =  self.tensor_list_.getInputTensor(4).getData()

        np_rpn_rois = self.tensor_list_.getOutputTensor(0)
        np_rpn_roi_probs = self.tensor_list_.getOutputTensor(1)
        np_rpn_rois_num = self.tensor_list_.getOutputTensor(2)
        np_rpn_rois_batch_size = self.tensor_list_.getOutputTensor(3)

        if (dtype == DataType.FLOAT32):
            scores = paddle.to_tensor(np_scores, dtype=paddle.float32)
            bbox_deltas = paddle.to_tensor(np_bboox_deltas, dtype=paddle.float32)
            anchors = paddle.to_tensor(np_anchors, dtype=paddle.float32)
            variances = paddle.to_tensor(np_variances, dtype=paddle.float32)
            img_size = paddle.to_tensor(np_img_size, dtype=paddle.float32)
            rpn_rois, rpn_roi_probs, rpn_rois_num = paddle.vision.ops.generate_proposals(
                        scores,
                        bbox_deltas,
                        img_size,
                        anchors,
                        variances,
                        self.pre_nms_top_n,
                        self.post_nms_top_n,
                        self.nms_thresh,
                        self.min_size,
                        self.eta,
                        self.pixel_offset,
                        return_rois_num=True)
            rpn_rois_tmp = rpn_rois.numpy()
            print(rpn_rois.shape[0])
            print(self.post_nms_top_n)
            add_zero_count = self.post_nms_top_n - rpn_rois.shape[0]
            rpn_rois_tmp = rpn_rois_tmp.reshape(rpn_rois_tmp.shape[0]*rpn_rois_tmp.shape[1])
            rpn_rois_tmp = np.pad(rpn_rois_tmp,(0, add_zero_count*4),'constant',constant_values=(0,0))
            rpn_rois_tmp = rpn_rois_tmp.reshape(self.post_nms_top_n, 4)
            np_rpn_rois.setData(rpn_rois_tmp)

            rpn_roi_probs_tmp = rpn_roi_probs.numpy()
            rpn_roi_probs_tmp = rpn_roi_probs_tmp.reshape(rpn_roi_probs_tmp.shape[0]*rpn_roi_probs_tmp.shape[1])
            rpn_roi_probs_tmp = np.pad(rpn_roi_probs_tmp,(0, add_zero_count),'constant',constant_values=(0,0))
            rpn_roi_probs_tmp = rpn_roi_probs_tmp.reshape(self.post_nms_top_n, 1)
            np_rpn_roi_probs.setData(rpn_roi_probs_tmp)

            np_rpn_rois_num.setData(rpn_rois_num.numpy())
            rpn_rois_batch_size_tmp = np.sum(rpn_rois_num.numpy())

            np_rpn_rois_batch_size.setData(np.array([rpn_rois_batch_size_tmp]))

            np_rpn_rois.setDiff(diff1=0.003, diff2=0.003)
            np_rpn_roi_probs.setDiff(diff3=0)
            np_rpn_rois_num.setDiff(diff3=0)
            np_rpn_rois_batch_size.setDiff(diff3=0)

            print("rpn_rois_batch_size", rpn_rois_batch_size_tmp)
        else:
           raise Exception("generate_proposals_v2 DataType should be Float, vs ", dtype)

@registerProtoWriter('generate_proposals_v2')
class OpTensorProtoWriter(MluOpProtoWriter):
    def dumpOpParam2Node(self):
        param_node = self.proto_node_.generate_proposals_v2_param
        param_node.pre_nms_top_n = self.op_params_.get("pre_nms_top_n", 6000)
        param_node.post_nms_top_n = self.op_params_.get("post_nms_top_n", 2000)
        param_node.nms_thresh = self.op_params_.get("nms_thresh", 0.5)
        param_node.min_size = self.op_params_.get("min_size", 0.1)
        param_node.eta = self.op_params_.get("eta", 1.0)
        param_node.pixel_offset = self.op_params_.get("pixel_offset", True)
