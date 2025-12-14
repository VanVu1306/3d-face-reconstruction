# postprocessing/improvements.py
import numpy as np
import cv2

def enhance_front_colors(colors: np.ndarray) -> np.ndarray:
    if len(colors) == 0:
        return colors

    size = int(np.sqrt(len(colors)))
    if size * size == len(colors):
        tex_img = colors.reshape(size, size, 3)

        for c in range(3):
            channel = tex_img[:, :, c]
            mean_val = np.mean(channel)
            if 50 < mean_val < 200:
                target = 128.0
                scale = target / mean_val if mean_val > 0 else 1.0
                tex_img[:, :, c] = np.clip(channel * scale, 0, 255)

        lab_img = cv2.cvtColor(tex_img.astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab_img)

        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
        l = clahe.apply(l)

        lab_img = cv2.merge([l, a, b])
        tex_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB)

        tex_img = cv2.bilateralFilter(tex_img, 3, 15, 15)

        return tex_img.reshape(-1, 3)

    return colors
