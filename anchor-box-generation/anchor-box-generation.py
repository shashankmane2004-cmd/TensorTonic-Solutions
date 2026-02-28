import numpy as np

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    Returns list of [x1, y1, x2, y2].
    """
    anchors = []
    stride = image_size / feature_size

    for i in range(feature_size):          # row
        for j in range(feature_size):      # col
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride

            for s in scales:
                for r in aspect_ratios:
                    w = s * np.sqrt(r)
                    h = s / np.sqrt(r)

                    x1 = cx - w / 2.0
                    y1 = cy - h / 2.0
                    x2 = cx + w / 2.0
                    y2 = cy + h / 2.0

                    anchors.append([x1, y1, x2, y2])

    return anchors