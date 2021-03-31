from utils.read_tif import read_bgrnir_tif_as_bgr
from models.tone_mapping import ToneMapping
from models.bertalmio import Bertalmio
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from models.color_balance import minmax_color_balance
from models.color_balance import simplest_color_balance_per_channel, simplest_color_balance
from models.retinex import MultiScaleRetinexHSV
from utils.histograms import bgr_histograms


def transfer_histogram(desired, image):
    out = np.zeros(image.shape, dtype=np.uint8)

    for i in range(3):
        hist_desired = np.histogram(desired[:, :, i], density=True, bins=256, range=(0, 255))[0].cumsum()
        hist_image = np.histogram(image[:, :, i], density=True, bins=256, range=(0, 255))[0].cumsum()

        for gray_level in range(256):
            s = hist_image[gray_level]
            new_gray = np.abs(s - hist_desired).argmin()

            mask = new_gray * (image[:, :, i] == gray_level)
            out[:, :, i] = mask

    return out


if __name__ == '__main__':
    # path = '/home/nviolante/projects/retinex/data/DT_RGB_enhancement_samples/20210203_000227_SN16_L1_Sydney-Australia/20210203_000227_SN16_L1_MS_Sydney-Australia_showcase_Full.tif'
    # path = '/home/nviolante/projects/retinex/data/DT_RGB_enhancement_samples/20210203_074054_SN16_L1_Balbala-Djibouti/20210203_074054_SN16_L1_MS_Balbala-بلبالا-Djibouti_showcase_Full.tif'
    # path = '/home/nviolante/projects/retinex/data/DT_RGB_enhancement_samples/20210203_173743_SN11_L1_Sterling-County-United-States/20210203_173743_SN11_L1_MS_Sterling-County-United-States_c-showcase_Full.tif'
    path = '/home/nviolante/projects/retinex/data/DT_RGB_enhancement_samples/20210210_031657_SN8_L1_Tachileik-Township-Myanmar/20210210_031657_SN8_L1_MS_Tachileik-Township-Myanmar_c-showcase_Full.tif'
    image = read_bgrnir_tif_as_bgr(path)
    tm = ToneMapping('reinhard', **{'gamma': 1.})
    # tm = Bertalmio(n=0.5)
    out = tm.apply(image)
    color = simplest_color_balance(out, 99.99, 1, new_min=50)

    # Y_image = minmax_color_balance(0.299*image[:,:,2].astype(np.float32) +0.587*image[:,:,1].astype(np.float32)+0.114*image[:,:,0].astype(np.float32), dtype=np.float32)
    # Y_color =0.299*color[:,:,2].astype(np.float32) +0.587*color[:,:,1].astype(np.float32)+0.114*color[:,:,0].astype(np.float32)
    # Y_out =0.299*out[:,:,2].astype(np.float32) +0.587*out[:,:,1].astype(np.float32)+0.114*out[:,:,0].astype(np.float32)
    # plt.figure()
    # plt.subplot(131)
    # plt.imshow(Y_image, cmap='gray')
    # plt.subplot(132)
    # plt.imshow(Y_out, cmap='gray')
    # plt.subplot(133)
    # plt.imshow(Y_color, cmap='gray')
    # plt.show()
    # print()

    # new = (Y_color/Y_image)
    # new[np.isnan(new)] = 0
    ret = MultiScaleRetinexHSV(scales=[15, 250, 450, 650], color_weight=0.5, output_weight=0.5)
    msr = ret.apply(out)
    plt.figure()
    plt.imshow(cv.cvtColor(simplest_color_balance(msr, 99.99, 1, new_min=30), cv.COLOR_BGR2RGB))
    plt.show()
    print()
    # ref = cv.imread('/home/nviolante/projects/retinex/data_old/google.png')
    # prueba = transfer_histogram(ref, out)

    # fig_image = plot_bgr_histograms(image)
    # fig_image.axes[0].set(title='Original')
    # fig_out = plot_bgr_histograms(out)
    # fig_out.axes[0].set(title='Tone Mapping')
    # fig_color = plot_bgr_histograms(color)
    # fig_color.axes[0].set(title='Tone Mapping + Stretching')

    # plt.figure()
    # plt.imshow(cv.cvtColor(minmax_color_balance(image), cv.COLOR_BGR2RGB))

    plt.figure()
    plt.imshow(cv.cvtColor(color, cv.COLOR_BGR2RGB))

    plt.show()
    print()
