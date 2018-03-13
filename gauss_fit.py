import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from process import *
from Hadamard import decode_img
from skimage.transform import rotate
from opencavity.beams import LgBasis
from opencavity.beams import HgBasis
#from scipy.stsci.image import translate
from skimage.transform import rotate, SimilarityTransform, warp

def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

class mode():
    def __init__(self,m,n):
        self.m = m
        self.l = n
    def L_mode(self,cord,amplitude,xo,yo,sigma_x,theta,offset):
        x = self.x
        y = self.y
        H=LgBasis(1,sigma_x)
        z=0.000000000000000001
        tem00=H.generate_lg(self.m, self.l, x, y, z)
        tem00 = np.abs(tem00)
        tem00 = rotate(tem00,theta)
        tform = SimilarityTransform(scale=offset, rotation=theta,translation=(xo, yo))
        tem00 = warp(tem00,tform)
        #tem00 = translate(tem00,x0,y0,mode='constant')
        tem00 = tem00*amplitude
        return tem00.ravel()
    def H_mode(self,cord,amplitude,xo,yo,sigma_x,theta,offset):
        x = self.x
        y = self.y
        H=HgBasis(1,sigma_x,sigma_x)
        z=0.000000000000000001
        tem00=H.generate_hg(self.m, self.l, x, y, z)
        tem00 = np.abs(tem00) 
        tform = SimilarityTransform(scale=offset, rotation=theta,translation=(xo, yo))
        tem00 = warp(tem00,tform)
        #tem00 = translate(tem00,x0,y0,mode='constant')
        tem00 = tem00*amplitude
        return tem00.ravel()

class opt_img():
    def __init__(self,image):
        self.image=image
        self.amp = True
    def opt(self,cord,amplitude,xo,yo,theta,offset):
        tform = SimilarityTransform(scale=offset, rotation=theta,translation=(xo, yo))
        t_image = warp(self.image,tform)
        #tem00 = translate(tem00,x0,y0,mode='constant')
        if self.amp:
            t_image = t_image*amplitude
        return t_image.ravel()

def fit_img(image,theory,initial_guess=[1,0,0,0,1]):
    fit_to = theory.ravel()
    to_fit = opt_img(image)
    funct = to_fit.opt
    x = np.linspace(image.shape[1],image.shape[1], image.shape[1])
    y = np.linspace(image.shape[1],image.shape[1], image.shape[0])
    x, y = np.meshgrid(x, y)
    popt, pcov = opt.curve_fit(funct, (x, y), fit_to, p0=initial_guess,maxfev=10000)
    to_fit.amp=False
    data_fitted = funct((x, y), *popt)
    return data_fitted.reshape(image.shape[0], image.shape[1]), popt, pcov




def fit_mode(image, initial_guess, basis,m,n):
    data = image.ravel()
    # Create x and y indices
    x = np.linspace(-initial_guess[3]*3, initial_guess[3]*3, image.shape[1])
    y = np.linspace(-initial_guess[3]*3, initial_guess[3]*3, image.shape[0])
    x, y = np.meshgrid(x, y)

    #plt.figure()
    #plt.imshow(data.reshape(image.shape[0], image.shape[1]))
    #plt.colorbar()
    if basis == 'L':
        t = mode(m,n)
        func = t.L_mode
        t.x = x
        t.y=y
    else:
        t = mode(m,n)
        func = t.H_mode
        t.x=x
        t.y=y

    #data_noisy = data + 0.2*np.random.normal(size=image.shape)
    bounds = [[50,-50,-50,30,-0.5,0.5],[300,50,50,100,0.5,2]]

    popt, pcov = opt.curve_fit(func, (x, y), data, p0=initial_guess,maxfev=10000,method='trf',bounds=bounds)

    data_fitted = func((x, y), *popt)
    #fig, ax = plt.subplots(1, 1)
    #ax.hold(True)
    #ax.imshow(data.reshape(image.shape[0], image.shape[1]), cmap=plt.cm.jet, origin='bottom',
    #    extent=(x.min(), x.max(), y.min(), y.max()))
    #ax.contour(x, y, data_fitted.reshape(image.shape[0], image.shape[1]), 8, colors='w')
    #data_fitted = func((x, y), *popt)
    return (0, 0), data_fitted.reshape(image.shape[0], image.shape[1]), popt, pcov



def fit_gauss(image, initial_guess=0):
    data = image.ravel()
    # Create x and y indices
    x = np.linspace(0, image.shape[1]+1, image.shape[1])
    y = np.linspace(0, image.shape[0]+1, image.shape[0])
    x, y = np.meshgrid(x, y)

    #create data
    #data = twoD_Gaussian((x, y), 3, 100, 100, 20, 40, 0, 10)

    # plot twoD_Gaussian data generated above
    plt.figure()
    plt.imshow(data.reshape(image.shape[0], image.shape[1]))
    plt.colorbar()

    # add some noise to the data and try to fit the data generated beforehand
    if initial_guess == 0:
        initial_guess = (3,100,100,20,40,0,10)
    else:
         pass
    #data_noisy = data + 0.2*np.random.normal(size=image.shape)
    popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data, p0=initial_guess,maxfev=10000,method='lm')

    data_fitted = twoD_Gaussian((x, y), *popt)
    fig, ax = plt.subplots(1, 1)
    ax.hold(True)
    ax.imshow(data.reshape(image.shape[0], image.shape[1]), cmap=plt.cm.jet, origin='bottom',
        extent=(x.min(), x.max(), y.min(), y.max()))
    ax.contour(x, y, data_fitted.reshape(image.shape[0], image.shape[1]), 8, colors='w')
    data_fitted = twoD_Gaussian((x, y), *popt)
    return (fig, ax), data_fitted.reshape(image.shape[0], image.shape[1]), popt, pcov


def moments(data,circle,rotate,vheight,estimator=np.ma.median,**kwargs):
    """Returns (height, amplitude, x, y, width_x, width_y, rotation angle)
    the gaussian parameters of a 2D distribution by calculating its
    moments.  Depending on the input parameters, will only output 
    a subset of the above.
    
    If using masked arrays, pass estimator=np.ma.median
    """
    total = np.abs(data).sum()
    Y, X = np.indices(data.shape) # python convention: reverse x,y np.indices
    y = np.argmax((X*np.abs(data)).sum(axis=1)/total)
    x = np.argmax((Y*np.abs(data)).sum(axis=0)/total)
    col = data[int(y),:]
    # FIRST moment, not second!
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)*col).sum()/np.abs(col).sum())
    row = data[:, int(x)]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)*row).sum()/np.abs(row).sum())
    width = ( width_x + width_y ) / 2.
    height = estimator(data.ravel())
    amplitude = data.max()-height
    mylist = [amplitude,x,y]
    if np.isnan(width_y) or np.isnan(width_x) or np.isnan(height) or np.isnan(amplitude):
        raise ValueError("something is nan")
    if vheight==1:
        mylist = [height] + mylist
    if circle==0:
        mylist = mylist + [width_x,width_y]
        if rotate==1:
            mylist = mylist + [0.] #rotation "moment" is just zero...
            # also, circles don't rotate.
    else:  
        mylist = mylist + [width]
    return mylist

