import cv2
import torch
import numpy as np
from lib.model import init_scrfd_500m_model


def preprocess(bgr_img, device):
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, device=device).unsqueeze(0).float()

    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.50196, 0.50196, 0.50196], device=device).view(1, 3, 1, 1)

    img /= 255.0
    img = (img - mean) / std
    return img


if __name__ == '__main__':
    device = 'cuda:0'
    img_path = '/app/assets/img.png'
    model = init_scrfd_500m_model(load_from='/app/weights/scrfd_500m_kps.pth', device=device)
    model.eval()

    bgr_img = cv2.imread(img_path)
    x = preprocess(bgr_img, device)

    with torch.no_grad():
        raw_output = model(x)
        result = model.postprocess(raw_output, conf_thresh=0.4, iou_thresh=0.45)

    scores, labels, bboxes, kps = result

    color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    vis_img = bgr_img.copy()
    for s, l, b, kp in zip(scores[0], labels[0], bboxes[0], kps[0]):
        x1, y1, x2, y2 = b.int()

        vis_img = cv2.rectangle(vis_img, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=1)
        cv2.putText(vis_img, f'Score: {s:.2f}', (x1, y1), color=color, fontFace=font, fontScale=1, thickness=1)

        min_side = min(x2-x1, y2-y1)
        marker_size = max(1, int(min_side / 20))
        for x, y in kp.view(5, 2):
            x, y = int(x), int(y)
            vis_img = cv2.circle(vis_img, center=(x, y), radius=marker_size, color=color, thickness=-1)

    cv2.imwrite('/app/result_img.jpg', vis_img)
