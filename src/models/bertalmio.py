import numpy as np
from models.color_balance import minmax_color_balance
from models.template import EnhancementStage
import cv2 as cv

class Bertalmio(EnhancementStage):
    """
    Tone mapping algorithm for HDR images based on perceptual principles and fine-tuned by cinema professionals.
    It is divided in two main stages: 1) light adaptation and 2) contrast adaptation.

    Ref:
        Bertalmio et al. 'Vision models fine-tuned by cinema professionals for High Dynamic Range imaging in movies'
        https://link.springer.com/article/10.1007/s11042-020-09532-y

    """
    def __init__(self, n):
        self.n = n
        
    def apply(self, image, output_path=None):
        y = 0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]  # BGR image
        y = minmax_color_balance(y, new_max=1.0, dtype=np.float32)

        n_bins = 1000
        H, X1 = np.histogram(y, bins=n_bins, density=True)
        idx = np.argmax(X1 > np.median(y))
        F1 = np.cumsum(H)
        F1 = F1 / F1.max()

        gamma_minus, _ = np.polyfit(X1[0:idx - 10], F1[0:idx - 10], 1)
        gamma_plus, _ = np.polyfit(X1[idx + 10:1000], F1[idx + 10:1000], 1)

        mu = np.median(y)  # np.log(np.median(Y)) - 2.0
        output = np.zeros_like(image, dtype=np.float32)

        r = minmax_color_balance(image[:, :, 2], 1, np.float32)
        output[:, :, 2] = np.power(r, p(r, gamma_plus, gamma_minus, mu, self.n))
        g = minmax_color_balance(image[:, :, 1], 1, np.float32)
        output[:, :, 1] = np.power(g, p(g, gamma_plus, gamma_minus, mu, self.n))
        b = minmax_color_balance(image[:, :, 0], 1, np.float32)
        output[:, :, 0] = np.power(b, p(b, gamma_plus, gamma_minus, mu, self.n))
        
        if output_path:
            cv.imwrite(output_path, output)

        return output
        


def p(c,gamma_plus,gamma_minus,mu1,n=1.0):
    mu = mu1 #np.log(mu1) - 2

    c=np.where((c > mu1/100.0) | (c == 0),c,mu1/100.0)

    mun = np.sign(mu)*np.power(np.abs(mu),n)
    cn = np.power(c,n)
    return gamma_plus + (gamma_minus-gamma_plus)*(mun)/(cn+mun)

