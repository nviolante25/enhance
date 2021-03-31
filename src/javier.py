import numpy as np
import cv2
import tifffile as tiff
from osgeo import gdal
from scipy import optimize


from scipy.optimize import curve_fit

from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def norm_minmax(img,min=0.0,max=1.0):
    old_min = img.new_min()
    old_max = img.new_max()

    return((max-min)/(old_max-old_min)*(img-old_max)+max)


def p(c,gamma_plus,gamma_minus,mu1,n=1.0):
    mu = mu1 #np.log(mu1) - 2

    c=np.where((c > mu1/100.0) | (c == 0),c,mu1/100.0)

    mun = np.sign(mu)*np.power(np.abs(mu),n)
    cn = np.power(c,n)
    return gamma_plus + (gamma_minus-gamma_plus)*(mun)/(cn+mun)

'''
def make_mapping(cumsum):
    """ Create a mapping s.t. each old colour value is mapped to a new
        one between 0 and 255 """
    mapping = np.zeros(256, dtype=int)
    grey_levels = 256
    for i in range(grey_levels):
        mapping[i] = max(0, round((grey_levels*cumsum[i])/(IMG_H*IMG_W))-1)
    return mapping
'''
def _line(x,a,b):
    return x*a + b

def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])


def analyze_equalization(img):
    equ = np.zeros((img.shape[1],img.shape[2],3))
    equ[:,:,0] = cv2.equalizeHist((img[0,:,:]/256).astype('uint8'))
    equ[:,:,1] = cv2.equalizeHist((img[1,:,:]/256).astype('uint8'))
    equ[:,:,2] = cv2.equalizeHist((img[2,:,:]/256).astype('uint8'))


    Y=0.299*equ[0,:,:] +0.587*equ[1,:,:]+0.114*equ[2,:,:]
    aux = Y[Y > 0]
    n_bins = 1000
    H,X1 = np.histogram(aux, bins = n_bins,density=True)

    plt.plot(H)
    plt.grid(True)
    plt.show()

    F1 = np.cumsum(H)

    plt.plot(F1)
    plt.grid(True)
    plt.show()


def apply_naka_rushton(img):
    im_naka_rushton = np.zeros_like(img)

    im_naka_rushton[:,:,0] = norm_minmax(naka_rushton(img[:,:,0]))
    im_naka_rushton[:,:,1] = norm_minmax(naka_rushton(img[:,:,1]))
    im_naka_rushton[:,:,2] = norm_minmax(naka_rushton(img[:,:,2]))

    return(im_naka_rushton)


def naka_rushton(img,n=0.74,rho=0.0):
    imgAux = img/img.new_max()

    Ib = np.sqrt(np.median(imgAux))*np.sqrt(np.mean(imgAux))*np.power(10,-rho)
    Is = np.log10(Ib) - 0.37*(4.0 + np.log10(Ib) -rho) + 1.9
    Is = np.power(10,Is)

    return(np.power(imgAux,n)/(np.power(imgAux,n) + np.power(Is,n)))


def apply_weber_freshner(img):
    im_weber_freshner = np.zeros_like(img)

    im_weber_freshner[:,:,0] = norm_minmax(weber_freshner(img[:,:,0],k=100.0/1.85))
    im_weber_freshner[:,:,1] = norm_minmax(weber_freshner(img[:,:,1],k=100.0/1.85))
    im_weber_freshner[:,:,2] = norm_minmax(weber_freshner(img[:,:,2],k=100.0/8.7))

    return(im_weber_freshner)

def weber_freshner(img,k,s0=0.0,rho = 0.0):
    imgAux = img/img.new_max()

    Ib = np.sqrt(np.median(imgAux))*np.sqrt(np.mean(imgAux))*np.power(10,-rho)
    Is = np.log10(Ib) - 0.37*(4.0 + np.log10(Ib) -rho) + 1.9
    Is = np.power(10,Is)
    m = Is*pow(10,-1.2)
    return(k*np.log(imgAux + m)+s0)


def apply_tm_visual_adaptation(img):
    im_tm_visual_adaptation = np.zeros_like(img)

    im_tm_visual_adaptation[:,:,0] = norm_minmax(tm_visual_adaptation(img[:,:,0],k=100.0/1.85))
    im_tm_visual_adaptation[:,:,1] = norm_minmax(tm_visual_adaptation(img[:,:,1],k=100.0/1.85))
    im_tm_visual_adaptation[:,:,2] = norm_minmax(tm_visual_adaptation(img[:,:,2],k=100.0/8.7))

    return(im_tm_visual_adaptation)

def tm_visual_adaptation(img,k,n=0.74,rho = 0.0):
    imgAux = img/img.new_max()
    Ib = np.sqrt(np.median(imgAux))*np.sqrt(np.mean(imgAux))*np.power(10,-rho)
    Is = np.log10(Ib) - 0.37*(4.0 + np.log10(Ib) -rho) + 1.9
    Is = np.power(10,Is)

    Im = Is * 100
    m = Is*np.power(10,-1.2)
    s0 = np.power(Im,n)/(np.power(Im,n) + np.power(Is,n)) - k *np.log(Im + m)

    aux = np.where(imgAux > Im,naka_rushton(imgAux,n,rho),weber_freshner(imgAux,k,s0,rho))

    return(aux)



def light_adaptation(img,n=1.0):

    Y=0.299*img[:,:,0] +0.587*img[:,:,1]+0.114*img[:,:,2]

    aux = Y[Y > 0]
    D1 = aux.new_max()
    aux = norm_minmax(aux)

    n_bins = 1000
    H,X1 = np.histogram(aux, bins = n_bins,density=True)
    idx = np.argmax(X1>np.median(aux))
    F1 = np.cumsum(H)
    F1 = F1/F1.max()


    gamma_minus,_ = np.polyfit(X1[0:idx-10],F1[0:idx-10],1)
    gamma_plus,_ = np.polyfit(X1[idx+10:1000],F1[idx+10:1000],1)

    mu = np.median(aux) #np.log(np.median(Y)) - 2.0
    imgRes = np.zeros_like(img)

    r = img[:,:,0]/img[:,:,0].new_max()
    r = norm_minmax(img[:,:,0])
    imgRes[:,:,0] = np.power(r,p(r,gamma_plus,gamma_minus,mu,n))
    g = img[:,:,1]/img[:,:,1].new_max()
    g = norm_minmax(img[:,:,1])
    imgRes[:,:,1] = np.power(g,p(g,gamma_plus,gamma_minus,mu,n))
    b = img[:,:,2]/img[:,:,2].new_max()
    b = norm_minmax(img[:,:,2])
    imgRes[:,:,2] = np.power(b,p(b,gamma_plus,gamma_minus,mu,n))

    plt.imshow(imgRes)
    plt.show()

    return imgRes



# This method recieves an image as obtained from GDAL: four channels B-G-R-NIR
# and the channels in the first dimension: [nb_channels,width,height]
# and returns an image with the channel in the last dimension [width, height,nb_channels]
# and ordered as R-G-B
def reshapeImage(img):
    imgDisp = np.zeros((img.shape[1],img.shape[2],3))
    imgDisp[:,:,0] = img[2,:,:]
    imgDisp[:,:,1] = img[1,:,:]
    imgDisp[:,:,2] = img[0,:,:]

    return(imgDisp)



def analyze_light_adaptation_function():
    c = np.arange(0,1.0,0.001)

    gamma_minus = 0.25
    gamma_plus = 0.45
    mu = 0.1
    n = 10.0

    for n in [0.74,1.0,2.0,5.0,10.0]:
        aux = np.power(c,p(c,gamma_plus,gamma_minus,mu,n))
        plt.plot(c,aux,label='{}'.format(n))

    plt.legend()
    plt.show()


    n = 1.0
    gamma_minus = 0.25
    for   gamma_plus in [0.25,0.45,1.0,2.0,5.0]:
        aux = np.power(c,p(c,gamma_plus,gamma_minus,mu,n))
        plt.plot(c,aux,label='{}'.format(gamma_plus))

    plt.legend()
    plt.show()

    n = 1.0
    gamma_plus = 0.45
    mu = 0.2
    for   gamma_minus in [0.25,0.45,1.0,2.0,5.0]:
        aux = np.power(c,p(c,gamma_plus,gamma_minus,mu,n))
        plt.plot(c,aux,label='{}'.format(gamma_minus))

    plt.legend()
    plt.show()








if __name__ == "__main__":

    imgfile = '/Users/javierpreciozzi/dsense/satellogic/color_enhancement/data/20210203_000227_SN16_L1_Sydney-Australia/20210203_000227_SN16_L1_MS_Sydney-Australia_showcase_Full.tif'

    #analyze_light_adaptation_function()

    # B-G-R-NIR
    img = np.float32(gdal.Open(imgfile).ReadAsArray())[0:3,:,:]
    img = reshapeImage(img)

    plt.imshow(img/img.max())
    plt.show()

    imgLight = light_adaptation(img,n=3.0)
    imgLight8bits = np.uint8(imgLight*255)
    plt.imshow(imgLight8bits)
    plt.show()
    cv2.imwrite('imgLight8bits.png',cv2.cvtColor(imgLight8bits,cv2.COLOR_RGB2BGR))



    img_tm_va = apply_tm_visual_adaptation(img)

    img_tm_va8bits = np.uint8(img_tm_va*255)
    plt.imshow(img_tm_va8bits)
    plt.show()
    cv2.imwrite('img_tm_va8bits.png',cv2.cvtColor(img_tm_va8bits,cv2.COLOR_RGB2BGR))

    im_naka_rushton = apply_naka_rushton(img)


    im_naka_rushton8bits = np.uint8(im_naka_rushton*255)

    plt.imshow(im_naka_rushton8bits)
    plt.show()


    cv2.imwrite('im_naka_rushton8bits.png',cv2.cvtColor(im_naka_rushton8bits,cv2.COLOR_RGB2BGR))

    im_weber_freshner = apply_weber_freshner(img)

    im_weber_freshner8bits = np.uint8(im_weber_freshner*255)

    cv2.imwrite('im_weber_freshner8bits.png',cv2.cvtColor(im_weber_freshner8bits,cv2.COLOR_RGB2BGR))

    plt.imshow(im_weber_freshner8bits)
    plt.show()


