import numpy as np
import cv2
import os
from typing import Tuple
import io
import tvm
import tvm.relay
import time
import onnx
import torch
import torchvision
import torch.onnx
from PIL import Image, ImageOps
import tvm.contrib.graph_runtime as graph_runtime
# from online_demo.mobilenet_v2_tsm import MobileNetV2
from mobilenet_v2_tsm import MobileNetV2

from models import TSN

import time

SOFTMAX_THRES = 0
HISTORY_LOGIT = True
REFINE_OUTPUT = True

pt_path = '../checkpoint/TSM_HockeyFights_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100/ckpt.best.pth.tar'


def torch2tvm_module(torch_module: torch.nn.Module, torch_inputs: Tuple[torch.Tensor, ...], target):
    torch_module.eval()
    input_names = []
    input_shapes = {}
    with torch.no_grad():
        for index, torch_input in enumerate(torch_inputs):
            name = "i" + str(index)
            input_names.append(name)
            input_shapes[name] = torch_input.shape

        # buffer = io.BytesIO()
        # torch.onnx.export(torch_module, torch_inputs, buffer, input_names=input_names,
        #                   # output_names=["o" + str(i) for i in range(len(torch_inputs))], opset_version=10)
        #                   output_names=["o" + str(i) for i in range(len(torch_inputs))], opset_version=10)
        # outs = torch_module(*torch_inputs)
        # buffer.seek(0, 0)
        # onnx_model = onnx.load_model(buffer)

        # print('torch_inputs:', len(torch_inputs), type(torch_inputs), input_names)
        # print(["o" + str(i) for i in range(len(torch_inputs))])

        torch.onnx.export(torch_module,  # model being run
                          torch_inputs[0],  # model input (or a tuple for multiple inputs)
                          "super_resolution.onnx",  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=11,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'])
        outs = torch_module(*torch_inputs)
        onnx_model = onnx.load_model("super_resolution.onnx")

        from onnxsim import simplify
        onnx_model, success = simplify(onnx_model)  # this simplifier removes conversion bugs.
        assert success
        print('assert success')
        relay_module, params = tvm.relay.frontend.from_onnx(onnx_model, shape=input_shapes)
        print('tvm.relay.frontend.from_onnx')
    with tvm.relay.build_config(opt_level=3):
        graph, tvm_module, params = tvm.relay.build(relay_module, target, params=params)
        print('tvm.relay.build')
    return graph, tvm_module, params


def torch2executor(torch_module: torch.nn.Module, torch_inputs: Tuple[torch.Tensor, ...], target):
    # prefix = f"mobilenet_tsm_tvm_{target}"
    # prefix = f"mobilenetv2_tsm_tvm_{target}"
    # prefix = f"mobilenetv3_tsm_tvm_{target}"
    prefix = f"mobilenetv4_tsm_tvm_{target}"
    lib_fname = f'{prefix}.tar'
    graph_fname = f'{prefix}.json'
    params_fname = f'{prefix}.params'
    if os.path.exists(lib_fname) and os.path.exists(graph_fname) and os.path.exists(params_fname):
        with open(graph_fname, 'rt') as f:
            graph = f.read()
        tvm_module = tvm.module.load(lib_fname)
        params = tvm.relay.load_param_dict(bytearray(open(params_fname, 'rb').read()))
    else:
        graph, tvm_module, params = torch2tvm_module(torch_module, torch_inputs, target)
        tvm_module.export_library(lib_fname)
        with open(graph_fname, 'wt') as f:
            f.write(graph)
        with open(params_fname, 'wb') as f:
            f.write(tvm.relay.save_param_dict(params))

    ctx = tvm.gpu() if target.startswith('cuda') else tvm.cpu()
    graph_module = graph_runtime.create(graph, tvm_module, ctx)
    for pname, pvalue in params.items():
        graph_module.set_input(pname, pvalue)

    def executor(inputs: Tuple[tvm.nd.NDArray]):
        for index, value in enumerate(inputs):
            graph_module.set_input(index, value)
        graph_module.run()
        return tuple(graph_module.get_output(index) for index in range(len(inputs)))

    return executor, ctx


def get_executor(use_gpu=True):
    # torch_module = MobileNetV2(n_class=27)
    # if not os.path.exists("mobilenetv2_jester_online.pth.tar"):  # checkpoint not downloaded
    #     print('Downloading PyTorch checkpoint...')
    #     import urllib.request
    #     url = 'https://file.lzhu.me/projects/tsm/models/mobilenetv2_jester_online.pth.tar'
    #     urllib.request.urlretrieve(url, './mobilenetv2_jester_online.pth.tar')

    # torch_module = MobileNetV2(n_class=2)
    # # torch_module.load_state_dict(torch.load("ckpt.best.pth.tar"))
    # checkpoint = torch.load("ckpt.best.pth.tar")
    # checkpoint = checkpoint['state_dict']
    #
    # # base_dict = {('base_model.' + k).replace('base_model.fc', 'new_fc'): v for k, v in list(checkpoint.items())}
    # base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
    # replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
    #                 'base_model.classifier.bias': 'new_fc.bias',
    #                 }
    # # print(base_dict.keys())
    # for k in base_dict.keys():
    #     k_new = k.replace('.net', '').replace('base_model.', '')
    #     base_dict[k_new] = base_dict.pop(k)
    #     # print(k, type(k))
    #     # print('base_model.features.2.conv.4.running_var'.startswith('base_model.'))
    #     # print('base_model.features.2.conv.4.running_var'.replace('base_model.', ''))
    #     # # exit()
    #     # if k.startswith('base_model.'):
    #     #     k_new = k.replace('base_model.', '')
    #     #     print(k_new)
    #     #     base_dict[k_new] = base_dict.pop(k)
    # # for k, v in replace_dict.items():
    # #     if k in base_dict:
    # #         base_dict[v] = base_dict.pop(k)
    # # print("\n\n\n")
    # # print(base_dict.keys())
    # for k in base_dict.keys():
    #     k_new = k.replace('.net', '').replace('base_model.', '')
    #     base_dict[k_new] = base_dict.pop(k)
    # # print("\n\n\n")
    # # print(base_dict.keys())
    # base_dict['classifier.weight'] = base_dict.pop('new_fc.weight')
    # base_dict['classifier.bias'] = base_dict.pop('new_fc.bias')
    # # exit()
    # # for k in base_dict.keys():
    # #     print(k, type(k))
    # #     print('base_model.features.2.conv.4.running_var'.startswith('base_model.'))
    # #     print('base_model.features.2.conv.4.running_var'.replace('base_model.', ''))
    # #     # exit()
    # #     if k.startswith('base_model.'):
    # #         k_new = k.replace('base_model.', '')
    # #         print(k_new)
    # #         base_dict[k_new] = base_dict.pop(k)
    # # print("\n\n\n")
    # # print(base_dict.keys())
    # torch_module.load_state_dict(base_dict)
    # # torch_module = torch.nn.DataParallel(torch_module.cuda())
    # # torch_module = torch_module.cuda()
    # # torch_module.eval()

    torch_module = TSN(2, 1, 'RGB',
                       base_model='mobilenetv2',
                       consensus_type='avg',
                       img_feature_dim=256,
                       pretrain='imagenet',
                       # is_shift=False, shift_div=8, shift_place='blockres',
                       is_shift=True, shift_div=8, shift_place='blockres',
                       # non_local='_nl' in './checkpoint/TSM_HockeyFights_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100/ckpt.best.pth.tar',
                       non_local='_nl' in pt_path,
                       )
    checkpoint = torch.load(
        # './checkpoint/TSM_HockeyFights_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100/ckpt.best.pth.tar')
        pt_path)
    checkpoint = checkpoint['state_dict']

    # base_dict = {('base_model.' + k).replace('base_model.fc', 'new_fc'): v for k, v in list(checkpoint.items())}
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
    replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                    'base_model.classifier.bias': 'new_fc.bias',
                    }
    for k, v in replace_dict.items():
        if k in base_dict:
            base_dict[v] = base_dict.pop(k)
    torch_module.load_state_dict(base_dict)

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
    # torch_inputs = (torch.rand(1, 3, 224, 224),)
    torch_inputs = (torch.rand(1, 8, 3, 224, 224),)
    # torch_inputs = (torch.rand(1, 24, 224, 224),)
    # torch_inputs = (torch.rand(1, 3, 224, 224).cuda(),
    #                 torch.zeros([1, 3, 56, 56]).cuda(),
    #                 torch.zeros([1, 4, 28, 28]).cuda(),
    #                 torch.zeros([1, 4, 28, 28]).cuda(),
    #                 torch.zeros([1, 8, 14, 14]).cuda(),
    #                 torch.zeros([1, 8, 14, 14]).cuda(),
    #                 torch.zeros([1, 8, 14, 14]).cuda(),
    #                 torch.zeros([1, 12, 14, 14]).cuda(),
    #                 torch.zeros([1, 12, 14, 14]).cuda(),
    #                 torch.zeros([1, 20, 7, 7]).cuda(),
    #                 torch.zeros([1, 20, 7, 7]).cuda())
    if use_gpu:
        target = 'cuda'
    else:
        target = 'llvm -mcpu=cortex-a72 -target=armv7l-linux-gnueabihf'
    # target = 'llvm -mcpu=cortex-a72 -target=armv7l-linux-gnueabihf'
    return torch2executor(torch_module, torch_inputs, target)


def transform(frame: np.ndarray):
    # 480, 640, 3, 0 ~ 255
    frame = cv2.resize(frame, (224, 224))  # (224, 224, 3) 0 ~ 255
    frame = frame / 255.0  # (224, 224, 3) 0 ~ 1.0
    frame = np.transpose(frame, axes=[2, 0, 1])  # (3, 224, 224) 0 ~ 1.0
    frame = np.expand_dims(frame, axis=0)  # (1, 3, 480, 640) 0 ~ 1.0
    return frame


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


# catigories = [
#     "Doing other things",  # 0
#     "Drumming Fingers",  # 1
#     "No gesture",  # 2
#     "Pulling Hand In",  # 3
#     "Pulling Two Fingers In",  # 4
#     "Pushing Hand Away",  # 5
#     "Pushing Two Fingers Away",  # 6
#     "Rolling Hand Backward",  # 7
#     "Rolling Hand Forward",  # 8
#     "Shaking Hand",  # 9
#     "Sliding Two Fingers Down",  # 10
#     "Sliding Two Fingers Left",  # 11
#     "Sliding Two Fingers Right",  # 12
#     "Sliding Two Fingers Up",  # 13
#     "Stop Sign",  # 14
#     "Swiping Down",  # 15
#     "Swiping Left",  # 16
#     "Swiping Right",  # 17
#     "Swiping Up",  # 18
#     "Thumb Down",  # 19
#     "Thumb Up",  # 20
#     "Turning Hand Clockwise",  # 21
#     "Turning Hand Counterclockwise",  # 22
#     "Zooming In With Full Hand",  # 23
#     "Zooming In With Two Fingers",  # 24
#     "Zooming Out With Full Hand",  # 25
#     "Zooming Out With Two Fingers"  # 26
# ]

catigories = [
    "no_HockFight",  # 0
    "yes_HockFight"  # 1
]

n_still_frame = 0


def process_output(idx_, history):
    # idx_: the output of current frame
    # history: a list containing the history of predictions
    if not REFINE_OUTPUT:
        return idx_, history

    max_hist_len = 20  # max history buffer
    history.append(idx_)
    history = history[-max_hist_len:]

    return history[-1], history

    # print("history5:", history)
    # # idx_: the output of current frame
    # # history: a list containing the history of predictions
    # if not REFINE_OUTPUT:
    #     return idx_, history
    #
    # max_hist_len = 20  # max history buffer
    #
    # print("history6:", history)
    #
    # # mask out illegal action
    # if idx_ in [7, 8, 21, 22, 3]:
    #     idx_ = history[-1]
    #
    # print("history7:", history)
    #
    # # use only single no action class
    # if idx_ == 0:
    #     idx_ = 2
    #
    # print("history8:", history)
    #
    # # history smoothing
    # if idx_ != history[-1] and len(history) > 1:
    #     print("come in ...")
    #     if not (history[-1] == history[-2]):  # and history[-2] == history[-3]):
    #         idx_ = history[-1]
    #
    # print("history9:", history)
    #
    # history.append(idx_)
    # history = history[-max_hist_len:]
    #
    # return history[-1], history


WINDOW_NAME = 'Video Gesture Recognition'


def main():
    print("Open camera...")
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('no1_xvid.avi')
    # cap = cv2.VideoCapture('fi1_xvid.avi')

    print(cap)

    # set a lower resolution for speed up
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # env variables
    full_screen = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    t = None
    index = 0
    print("Build transformer...")
    transform = get_transform()
    print("Build Executor...")
    executor, ctx = get_executor()
    # executor, ctx = get_executor(use_gpu=False)
    # # buffer = (
    # #     tvm.nd.empty((1, 3, 56, 56), ctx=ctx),
    # #     tvm.nd.empty((1, 4, 28, 28), ctx=ctx),
    # #     tvm.nd.empty((1, 4, 28, 28), ctx=ctx),
    # #     tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
    # #     tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
    # #     tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
    # #     tvm.nd.empty((1, 12, 14, 14), ctx=ctx),
    # #     tvm.nd.empty((1, 12, 14, 14), ctx=ctx),
    # #     tvm.nd.empty((1, 20, 7, 7), ctx=ctx),
    # #     tvm.nd.empty((1, 20, 7, 7), ctx=ctx)
    # )
    idx = 0
    history = [2]
    history_logit = []
    history_timing = []

    i_frame = -1

    print("Ready!")
    # print("history1:", history)
    while True:
        i_frame += 1
        _, img = cap.read()  # (480, 640, 3) 0 ~ 255
        # print("history2:", history)
        # print(i_frame)
        t0 = time.time()
        if i_frame % 2 == 0:  # skip every other frame to obtain a suitable frame rate
            t1 = time.time()
            # print("type(img):", type(img), img.shape)  # (720, 1280, 3)
            img_tran = transform([Image.fromarray(img).convert('RGB')])
            # print("type(img_tran):", type(img_tran), img_tran.shape)  # torch.Size([3, 224, 224])
            input_var = torch.autograd.Variable(img_tran.view(1, 3, img_tran.size(1), img_tran.size(2)))
            # print("type(input_var):", type(input_var), input_var.shape)  # torch.Size([1, 3, 224, 224])
            img_nd = tvm.nd.array(input_var.detach().numpy(), ctx=ctx)
            # print("type(img_nd):", type(img_nd), img_nd.shape)  # (1, 3, 224, 224)

            # print("type(buffer):", type(buffer), len(buffer))  # (1, 3, 224, 224)
            # print(type(buffer[0]), buffer[0].shape)
            # print(type(buffer[1]), buffer[1].shape)

            # inputs: Tuple[tvm.nd.NDArray] = (img_nd,) + buffer
            inputs: Tuple[tvm.nd.NDArray] = (img_nd,)

            # print("type(inputs):", type(inputs), len(inputs))
            # print(type(inputs[0]), inputs[0].shape)
            # print(type(inputs[1]), inputs[1].shape)

            # t11 = time.time()
            # print("t11-t1:", t11 - t1)
            outputs = executor(inputs)
            # print("outputs:", type(outputs))
            # t12 = time.time()
            # print("t12-t11:", t12 - t11)

            # print("type(outputs):", type(outputs), len(outputs))  # (1, 3, 224, 224)
            # print(type(outputs[0]), outputs[0].shape)
            # print(type(outputs[1]), outputs[1].shape)

            # print("buffer before:", buffer[0].shape, type(buffer), type(buffer[0]))
            # buffer_before = buffer[0]

            # feat, buffer = outputs[0], outputs[1:]
            feat = outputs[0]

            # print("feat:", type(feat))
            # # print("feat:", len(feat))
            # t13 = time.time()
            # print("t13-t12:", t13 - t12)

            # print("buffer after:", buffer[0].shape, type(buffer), type(buffer[0]))
            # buffer_after = buffer[0]
            # print(buffer_before == buffer_after)
            # buffer_new = (
            #     tvm.nd.empty((1, 3, 56, 56), ctx=ctx),
            #     tvm.nd.empty((1, 4, 28, 28), ctx=ctx),
            #     tvm.nd.empty((1, 4, 28, 28), ctx=ctx),
            #     tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
            #     tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
            #     tvm.nd.empty((1, 8, 14, 14), ctx=ctx),
            #     tvm.nd.empty((1, 12, 14, 14), ctx=ctx),
            #     tvm.nd.empty((1, 12, 14, 14), ctx=ctx),
            #     tvm.nd.empty((1, 20, 7, 7), ctx=ctx),
            #     tvm.nd.empty((1, 20, 7, 7), ctx=ctx)
            # )
            # print(buffer_before == buffer_new)
            # print(buffer_after == buffer_new)

            print(type(feat))
            assert isinstance(feat, tvm.nd.NDArray)

            # print("SOFTMAX_THRES:", SOFTMAX_THRES)
            # if SOFTMAX_THRES > 0:
            #     print('if SOFTMAX_THRES > 0:')
            #     feat_np = feat.asnumpy().reshape(-1)
            #     feat_np -= feat_np.max()
            #     softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))
            #
            #     print(max(softmax))
            #     if max(softmax) > SOFTMAX_THRES:
            #         idx_ = np.argmax(feat.asnumpy(), axis=1)[0]
            #     else:
            #         idx_ = idx
            #     # print(idx_)
            # else:
            #     print(feat, type(feat))
            #     t18 = time.time()
            #     print("t18-t14:", t18 - t14)
            #     feat_numpy = feat.asnumpy()
            #     t19 = time.time()
            #     print("t19-t18:", t19 - t18)
            #     idx_ = np.argmax(feat_numpy, axis=1)[0]
            #     # idx_ = np.argmax(feat.asnumpy(), axis=1)[0]
            #     t20 = time.time()
            #     print("t20-t19:", t20 - t19)
            #     print("t20-t18:", t20 - t18)

            # t14 = time.time()
            # print("t14-t13:", t14 - t13)
            # print(feat[0], feat[0][0], feat[0][1])
            # print(feat)
            # t144 = time.time()
            # print("t144-t14:", t144 - t14)
            # print(feat.dtype, feat.context, feat.shape, feat.ctx)
            # print(feat_numpy.shape)
            t145 = time.time()
            # print("t145-t144:", t145 - t144)

            idx_ = np.argmax(feat.asnumpy(), axis=1)[0]
            t15 = time.time()
            print("t15-t145:", t15 - t145)
            # print("t15-t14:", t15 - t14)

            if HISTORY_LOGIT:
                history_logit.append(feat.asnumpy())
                history_logit = history_logit[-12:]
                avg_logit = sum(history_logit)
                idx_ = np.argmax(avg_logit, axis=1)[0]

            # t16 = time.time()
            # print("t16-t15:", t16 - t15)

            # print("----------")
            # print("history3:", history)
            # print(idx_, type(idx_))
            # print(history, type(history))
            idx, history = process_output(idx_, history)
            # print("history4:", history)

            t2 = time.time()
            # print("t2-t16:", t2 - t16)
            # print("t2-t12:", t2 - t12)
            # print("t2-t1:", t2 - t1)
            print(f"{index} {catigories[idx]}")

            current_time = t2 - t1

        # t4 = time.time()
        # print("t4-t1:", t4 - t1)
        img = cv2.resize(img, (640, 480))
        img = img[:, ::-1]
        height, width, _ = img.shape
        label = np.zeros([height // 10, width, 3]).astype('uint8') + 255

        cv2.putText(label, 'Prediction: ' + catigories[idx],
                    (0, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)
        cv2.putText(label, '{:.1f} Vid/s'.format(1 / current_time),
                    (width - 170, int(height / 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)

        img = np.concatenate((img, label), axis=0)
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:  # exit
            break
        elif key == ord('F') or key == ord('f'):  # full screen
            print('Changing full screen option!')
            full_screen = not full_screen
            if full_screen:
                print('Setting FS!!!')
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL)

        # t5 = time.time()
        # print("t5-t4:", t5 - t4)
        if t is None:
            t = time.time()
        else:
            nt = time.time()
            index += 1
            t = nt

    cap.release()
    cv2.destroyAllWindows()


main()
