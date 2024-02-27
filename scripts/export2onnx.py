import numpy as np
import onnx
import os
import torch


class OnnxEngine:
    """
    OnnxEngine
    """
    def __init__(
        self,
        model_path: str,
        num_threads = None,
        device: str = 'cpu',
    ):
        """
        :param model_path: path to the .onnx model
        :param num_threads: number of cpu cores to run inference on. Only used if device is set to 'cpu'
        :param device: device to run onnx inference on (cpu or cuda)
        """
        self.model_path = model_path

        if device == 'cpu':
            providers = ['CPUExecutionProvider']
        else:
            providers = ['CUDAExecutionProvider']

        # num threads in env should be set before onnxruntime import
        if device == 'cpu' and num_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(num_threads)
        import onnxruntime

        sess_options = onnxruntime.SessionOptions()
        if num_threads is not None:
            sess_options.intra_op_num_threads = num_threads
            sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

        self.session = onnxruntime.InferenceSession(self.model_path, sess_options, providers=providers)
        self.input_name = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        if num_threads is not None:
            print(f'Device: {device}, NumThreads: {num_threads}')
        else:
            print(f'Device: {device}')

    def dummy_forward(self, x: np.ndarray):
        inputs = {self.input_name[0]: x.astype(np.float32)}
        out = self.session.run(self.output_names, inputs)
        return out


def scrfd2onnx(
    model,
    input_shape,
    opset_version=11,
    save_file='tmp.onnx',
    simplify=True,
    dynamic=True,
    verbose=True,
):
    tensor_data = torch.rand((1, 3, *input_shape))

    model.eval()

    # Define input and outputs names, which are required to properly define
    # dynamic axes
    input_names = ['input.1']
    output_names = [
        'score_8', 'score_16', 'score_32',
        'bbox_8', 'bbox_16', 'bbox_32',
    ]

    # Define dynamic axes for export
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {out: {0: '?', 1: '?'} for out in output_names}
        dynamic_axes[input_names[0]] = {
            0: '?',
            2: '?',
            3: '?'
        }

    torch.onnx.export(
        model,
        tensor_data,
        save_file,
        keep_initializers_as_inputs=False,
        verbose=verbose,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version
    )

    if simplify:
        onnx_model = onnx.load(save_file)
        from onnxsim import simplify
        onnx_model, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(onnx_model, save_file)

    engine = OnnxEngine(save_file)

    x = torch.rand((1, 3, *input_shape))
    with torch.no_grad():
        res_pt = model(x)
        res_pt = [*res_pt[0], * res_pt[1]]

    res_onnx = engine.dummy_forward(x.numpy())

    for i, (x, y) in enumerate(zip(res_onnx, res_pt)):
        y = y.numpy()
        if not np.allclose(x, y):
            rate = np.mean(np.abs(x-y) / y)
            print(f'Output {i} ({output_names[i]}): {rate:.3e}')
            continue
        print(f'Output {i} ({output_names[i]}): no diff')

    print(f'Successfully exported ONNX model: {save_file}')


if __name__ == '__main__':
    from lib.model import init_scrfd_10g_model

    input_size = (640, 640)
    load_from = '/app/weights/scrfd_10g_kps.pth'
    model = init_scrfd_10g_model(load_from, device='cpu', use_kps=True)

    model.eval()
    scrfd2onnx(
        model,
        input_shape=input_size,
        opset_version=11,
        save_file=save_file,
        simplify=True,
        dynamic=False,
        verbose=True,
    )

    import cv2

    bgr_img = cv2.imread('/app/assets/img.png')
    bgr_img = cv2.resize(bgr_img, input_size)

    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = img / 255.0

    mean = np.array([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
    std = np.array([0.50196, 0.50196, 0.50196]).reshape(1, 3, 1, 1)

    img = (img - mean) / std
    img = img.astype(np.float32)
    img = torch.from_numpy(img)

    model.eval()
    with torch.no_grad():
        raw_res = model(img)

    results = model.postprocess(raw_res, conf_thresh=0.35, iou_thresh=0.45)
    for l, s, box in zip(results[0][0], results[1][0], results[2][0]):
        x1, y1, x2, y2 = box.int().numpy()
        bgr_img = cv2.rectangle(bgr_img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=1)
    cv2.imwrite('/app/result_torch.jpg', bgr_img)
