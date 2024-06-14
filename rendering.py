import cv2
import math
import argparse
import numpy as np
from matplotlib import pyplot as plt

class ColorBlindConverter(object):
    def __init__(self):
        (
            self.Normal,
            self.Protanopia,
            self.Deuteranopia,
            self.Tritanopia,
            self.Protanomaly,
            self.Deuteranomaly,
            self.Tritanomaly,
            self.Monochromacy
        ) = range(8)

        self.powGammaLookup = np.power(np.linspace(0, 256, 256) / 256, 2.2)
        self.conversion_coeffs = [
            {'cpu': 0.735, 'cpv':  0.265, 'am': 1.273463, 'ayi': -0.073894},
            {'cpu': 1.140, 'cpv': -0.140, 'am': 0.968437, 'ayi':  0.003331},
            {'cpu': 0.171, 'cpv': -0.003, 'am': 0.062921, 'ayi':  0.292119}]

    def _inversePow(self, x):
        return int(255.0 * float(0 if x <= 0 else (1 if x >= 1 else np.power(x, 1/2.2))))

    def convert(self, rgb, cb_type):
        self.rgb = rgb
        self.cb_type = cb_type
        
        if self.cb_type == 0:
            self.converted_rgb = self._convert_normal()
        elif self.cb_type in range(1, 4):
            self.converted_rgb = self._convert_colorblind()
        elif self.cb_type in range(4, 7):
            self.converted_rgb = self._convert_anomylize(self._convert_colorblind())
        elif self.cb_type == 7:
            self.converted_rgb = self._convert_monochrome()
        return self.converted_rgb

    def _convert_normal(self):
        return self.rgb

    def _convert_colorblind(self):
        wx = 0.312713
        wy = 0.329016
        wz = 0.358271

        cpu, cpv, am, ayi = self.conversion_coeffs[{
            1: 0, 4: 0,
            2: 1, 5: 1,
            3: 2, 6: 2,
        }[self.cb_type]].values()

        r, g, b = self.rgb

        cr = self.powGammaLookup[r]
        cg = self.powGammaLookup[g]
        cb = self.powGammaLookup[b]

        # rgb -> xyz
        cx = (0.430574 * cr + 0.341550 * cg + 0.178325 * cb)
        cy = (0.222015 * cr + 0.706655 * cg + 0.071330 * cb)
        cz = (0.020183 * cr + 0.129553 * cg + 0.939180 * cb)

        sum_xyz = cx + cy + cz
        cu = 0
        cv = 0

        if(sum_xyz != 0):
            cu = cx / sum_xyz
            cv = cy / sum_xyz

        nx = wx * cy / wy
        nz = wz * cy / wy
        clm = 0
        dy = 0

        clm = (cpv - cv) / (cpu - cu) if (cu < cpu) else (cv - cpv) / (cu - cpu)
        clyi = cv - cu * clm
        du = (ayi - clyi) / (clm - am)
        dv = (clm * du) + clyi

        sx = du * cy / dv
        sy = cy
        sz = (1 - (du + dv)) * cy / dv

        # xyz->rgb
        sr =  (3.063218 * sx - 1.393325 * sy - 0.475802 * sz)
        sg = (-0.969243 * sx + 1.875966 * sy + 0.041555 * sz)
        sb =  (0.067871 * sx - 0.228834 * sy + 1.069251 * sz)

        dx = nx - sx
        dz = nz - sz

        # xyz->rgb

        dr =  (3.063218 * dx - 1.393325 * dy - 0.475802 * dz)
        dg = (-0.969243 * dx + 1.875966 * dy + 0.041555 * dz)
        db =  (0.067871 * dx - 0.228834 * dy + 1.069251 * dz)

        adjr = ((0 if sr < 0 else 1) - sr) / dr if dr > 0 else 0
        adjg = ((0 if sg < 0 else 1) - sg) / dg if dg > 0 else 0
        adjb = ((0 if sb < 0 else 1) - sb) / db if db > 0 else 0

        adjust = max([
            0 if (adjr > 1 or adjr < 0) else adjr,
            0 if (adjg > 1 or adjg < 0) else adjg,
            0 if (adjb > 1 or adjb < 0) else adjb])

        sr = sr + (adjust * dr)
        sg = sg + (adjust * dg)
        sb = sb + (adjust * db)

        return [self._inversePow(sr), self._inversePow(sg), self._inversePow(sb)]

    def _convert_anomylize(self, p_cb):
        v = 1.75
        d = v + 1
        
        r_orig, g_orig, b_orig = self.rgb
        r_cb, g_cb, b_cb = p_cb

        r_new = (v * r_cb + r_orig) / d
        g_new = (v * g_cb + g_orig) / d
        b_new = (v * b_cb + b_orig) / d

        return [int(r_new), int(g_new), int(b_new)]

    def _convert_monochrome(self):
        r_old, g_old, b_old = self.rgb
        g_new = (r_old * 0.299) + (g_old * 0.587) + (b_old * 0.114)
        return [int(g_new)] * 3

def main(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image for each type of colorblindness
    cbconv = ColorBlindConverter()
    cb_types = [
        'Normal',
        'Protanopia',
        'Deuteranopia',
        'Tritanopia',
        'Protanomaly',
        'Deuteranomaly',
        'Tritanomaly',
        'Monochromacy'
    ]

    # Create a dictionary to store the converted images
    converted_images = {}

    for cb_type_name in cb_types:
        cb_type_index = cb_types.index(cb_type_name)
        converted_image = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                converted_image[i, j] = cbconv.convert(image[i, j], cb_type_index)
        converted_images[cb_type_name] = converted_image

    # Determine the number of rows and columns for subplots
    num_cb_types = len(cb_types)
    cols = 4
    rows = math.ceil(num_cb_types / cols)

    # Plot the original and converted images
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()
    for idx, cb_type_name in enumerate(cb_types):
        axes[idx].imshow(converted_images[cb_type_name])
        axes[idx].set_title(cb_type_name, fontsize=20)  # Increase the label size
        axes[idx].axis('off')

    # Hide any unused subplots
    for i in range(num_cb_types, rows * cols):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)  # Save the figure to the specified path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert image for different types of colorblindness")
    parser.add_argument("--image_path", type=str, default='./data/Capture1.PNG', help="Path to the input image")
    parser.add_argument("--output_path", type=str, default='./results/Capture1_rendering', help="Path to save the output image")
    args = parser.parse_args()

    main(args.image_path, args.output_path)
