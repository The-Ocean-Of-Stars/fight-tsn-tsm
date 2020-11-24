import cv2
import sys
import os
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import math
import time

from PIL import Image
import torch
import torchvision


class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        # self.worker = torchvision.transforms.Scale(size, interpolation)
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


def get_transform():
    cropping = torchvision.transforms.Compose([
        GroupScale(256),
        GroupCenterCrop(224),
    ])
    transform = torchvision.transforms.Compose([
        cropping,
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform


class Processor():
    def __init__(self, model):
        # print('setting up Yolov5s-simple.trt processor')
        print('setting up mobilev2.trt processor')
        # load tensorrt engine
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        # print(os.path.dirname(__file__))
        TRTbin = '{0}/models/{1}'.format(os.path.dirname(__file__), model)
        # print(TRTbin)
        with open(TRTbin, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            # print(type(runtime))
            engine = runtime.deserialize_cuda_engine(f.read())
            # print(type(engine))
        self.context = engine.create_execution_context()
        # allocate memory
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        # save to class
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.stream = stream

        self.transform = get_transform()

        print('success')

        # # post processing config
        # filters = (80 + 5) * 3
        # self.output_shapes = [
        #     (1, 3, 80, 80, 85),
        #     (1, 3, 40, 40, 85),
        #     (1, 3, 20, 20, 85)
        # ]
        # self.strides = np.array([8., 16., 32.])
        # anchors = np.array([
        #     [[10,13], [16,30], [33,23]],
        #     [[30,61], [62,45], [59,119]],
        #     [[116,90], [156,198], [373,326]],
        # ])
        # self.nl = len(anchors)
        # self.nc = 80 # classes
        # self.no = self.nc + 5 # outputs per anchor
        # self.na = len(anchors[0])
        # a = anchors.copy().astype(np.float32)
        # a = a.reshape(self.nl, -1, 2)
        # self.anchors = a.copy()
        # self.anchor_grid = a.copy().reshape(self.nl, 1, -1, 1, 1, 2)

    def detect(self, img):
        # print("detect:")
        # print(img.shape, len(buffer))
        resized = self.pre_process(img)
        # print(resized.shape)

        # torch_inputs = (torch.rand(1, 3, 224, 224),
        #                 torch.zeros([1, 3, 56, 56]),
        #                 torch.zeros([1, 4, 28, 28]),
        #                 torch.zeros([1, 4, 28, 28]),
        #                 torch.zeros([1, 8, 14, 14]),
        #                 torch.zeros([1, 8, 14, 14]),
        #                 torch.zeros([1, 8, 14, 14]),
        #                 torch.zeros([1, 12, 14, 14]),
        #                 torch.zeros([1, 12, 14, 14]),
        #                 torch.zeros([1, 20, 7, 7]),
        #                 torch.zeros([1, 20, 7, 7]))
        # torch_inputs = (resized, *buffer)
        # print('torch_inputs:', len(torch_inputs), torch_inputs[0].shape, torch_inputs[1].shape)

        # outputs = self.inference(resized, buffer)
        outputs = self.inference(resized)
        # print(type(outputs), len(outputs))
        # print(outputs[-2])
        print(outputs[-1], type(outputs))
        # for output in outputs:
        #     print(output.shape)
        # reshape from flat to (1, 3, x, y, 85)
        # reshaped = []
        # for output, shape in zip(outputs, self.output_shapes):
        #     reshaped.append(output.reshape(shape))
        # return reshaped
        print(outputs[-1].argmax(axis=0))
        return outputs[-1].argmax(axis=0)

    def pre_process(self, img):
        # print('original image shape', img.shape)
        img_tran = self.transform([Image.fromarray(img).convert('RGB')])
        # print("type(img_tran):", type(img_tran), img_tran.shape)  # torch.Size([3, 224, 224])
        input_var = torch.autograd.Variable(img_tran.view(1, 3, img_tran.size(1), img_tran.size(2)))
        # print("type(input_var):", type(input_var), input_var.shape)  # torch.Size([1, 3, 224, 224])
        img_reshaped = input_var.detach().numpy()

        # print('x:', img_reshaped)

        return img_reshaped

    # def inference(self, img, buffer):
    def inference(self, torch_inputs):
        # print('inference')
        # copy img to input memory
        # self.inputs[0]['host'] = np.ascontiguousarray(img)
        # self.inputs[0]['host'] = np.ravel(img)
        self.inputs[0]['host'] = np.ravel(torch_inputs)
        # self.inputs[0]['host'] = np.ravel(torch_inputs[0])
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        start = time.time()
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        end = time.time()
        print('execution time:', end - start)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()
        return [out['host'] for out in self.outputs]

    def extract_object_grids(self, output):
        """
        Extract objectness grid 
        (how likely a box is to contain the center of a bounding box)
        Returns:
            object_grids: list of tensors (1, 3, nx, ny, 1)
        """
        object_grids = []
        for out in output:
            probs = self.sigmoid_v(out[..., 4:5])
            object_grids.append(probs)
        return object_grids

    def extract_class_grids(self, output):
        """
        Extracts class probabilities
        (the most likely class of a given tile)
        Returns:
            class_grids: array len 3 of tensors ( 1, 3, nx, ny, 80)
        """
        class_grids = []
        for out in output:
            object_probs = self.sigmoid_v(out[..., 4:5])
            class_probs = self.sigmoid_v(out[..., 5:])
            obj_class_probs = class_probs * object_probs
            class_grids.append(obj_class_probs)
        return class_grids

    def extract_boxes(self, output, conf_thres=0.5):
        """
        Extracts boxes (xywh) -> (x1, y1, x2, y2)
        """
        scaled = []
        grids = []
        for out in output:
            out = self.sigmoid_v(out)
            _, _, width, height, _ = out.shape
            grid = self.make_grid(width, height)
            grids.append(grid)
            scaled.append(out)
        z = []
        for out, grid, stride, anchor in zip(scaled, grids, self.strides, self.anchor_grid):
            _, _, width, height, _ = out.shape
            out[..., 0:2] = (out[..., 0:2] * 2. - 0.5 + grid) * stride
            out[..., 2:4] = (out[..., 2:4] * 2) ** 2 * anchor

            out[..., 5:] = out[..., 4:5] * out[..., 5:]
            out = out.reshape((1, 3 * width * height, 85))
            z.append(out)
        pred = np.concatenate(z, 1)
        xc = pred[..., 4] > conf_thres
        pred = pred[xc]
        boxes = self.xywh2xyxy(pred[:, :4])
        return boxes

    def post_process(self, outputs, conf_thres=0.5):
        """
        Transforms raw output into boxes, confs, classes
        Applies NMS thresholding on bounding boxes and confs
        Parameters:
            output: raw output tensor
        Returns:
            boxes: x1,y1,x2,y2 tensor (dets, 4)
            confs: class * obj prob tensor (dets, 1) 
            classes: class type tensor (dets, 1)
        """
        scaled = []
        grids = []
        for out in outputs:
            out = self.sigmoid_v(out)
            _, _, width, height, _ = out.shape
            grid = self.make_grid(width, height)
            grids.append(grid)
            scaled.append(out)
        z = []
        for out, grid, stride, anchor in zip(scaled, grids, self.strides, self.anchor_grid):
            _, _, width, height, _ = out.shape
            out[..., 0:2] = (out[..., 0:2] * 2. - 0.5 + grid) * stride
            out[..., 2:4] = (out[..., 2:4] * 2) ** 2 * anchor

            out = out.reshape((1, 3 * width * height, 85))
            z.append(out)
        pred = np.concatenate(z, 1)
        xc = pred[..., 4] > conf_thres
        pred = pred[xc]
        return self.nms(pred)

    def make_grid(self, nx, ny):
        """
        Create scaling tensor based on box location
        Source: https://github.com/ultralytics/yolov5/blob/master/models/yolo.py
        Arguments
            nx: x-axis num boxes
            ny: y-axis num boxes
        Returns
            grid: tensor of shape (1, 1, nx, ny, 80)
        """
        nx_vec = np.arange(nx)
        ny_vec = np.arange(ny)
        yv, xv = np.meshgrid(ny_vec, nx_vec)
        grid = np.stack((yv, xv), axis=2)
        grid = grid.reshape(1, 1, ny, nx, 2)
        return grid

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_v(self, array):
        return np.reciprocal(np.exp(-array) + 1.0)

    def exponential_v(self, array):
        return np.exp(array)

    def non_max_suppression(self, boxes, confs, classes, iou_thres=0.6):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = confs.flatten().argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thres)[0]
            order = order[inds + 1]
        boxes = boxes[keep]
        confs = confs[keep]
        classes = classes[keep]
        return boxes, confs, classes

    def nms(self, pred, iou_thres=0.6):
        boxes = self.xywh2xyxy(pred[..., 0:4])
        # best class only
        confs = np.amax(pred[:, 5:], 1, keepdims=True)
        classes = np.argmax(pred[:, 5:], axis=-1)
        return self.non_max_suppression(boxes, confs, classes)

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
