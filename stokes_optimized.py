from skimage import data, io, util
from skimage.color import label2rgb
from scipy.misc import imresize
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.linalg import sqrtm
import uncertainties
import os
from gauss_fit import fit_img
import theoretical as theory
from uncertainties import unumpy as unp
from finesse import proc_img
from matplotlib.colors import LinearSegmentedColormap
import pylab


folder = 'C:/Users/danie/Pictures/masters/stokes/'
folder = './stokes/stokes_2/'

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def bin_ndarray(ndarray, new_shape, operation='mean'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray


def import_image(filename,base_path,resize=False):
    if filename[-3:] == 'npy':
        if resize:
            image = np.load(base_path+filename)
            shape = image.shape
            new = np.zeros((shape[0],int(shape[1]/float(resize)),int((shape[2]/resize))))
            for i,val in enumerate(image):
                new[i] = bin_ndarray(val,(int(shape[1]/float(resize)),int(shape[2]/float(resize))))
            return new
        else:
            return np.load(base_path+filename)
    else:
        return io.imread(base_path+filename)
    

def image_uncertainty(image,crop=False,uncertainty=True):  
    if crop:
        image=image[:,crop[0]:crop[1],crop[2]:crop[3]] 
    if image.shape[0] < image.shape[1]:
        if uncertainty:
            return unp.uarray(image.mean(axis=0), image.std(axis=0))
        else:
            return image.mean(axis=0)

def collect_data(base_path):
    f = [f for f in os.listdir(base_path)]
    temp_sets = {}

    for i in f:
        if i[-3:] == 'npy':
            if not i.split('_')[0] in temp_sets:
                k=[]
                for j in f:
                    if i.split('_')[0] == j.split('_')[0]:
                        #j = image_uncertainty(import_image(j,base_path))
                        k.append(j)
                k = {'A':k[0],'D':k[1],'H':k[2],'L':k[3],'R':k[4],'V':k[5]}
                temp_sets[i.split('_')[0]]=k

    return temp_sets 

def process_data(entry,crop,resize,base_path,uncertainty=False):
    for i in entry:
            temp = import_image(entry[i],base_path,resize=resize)
            mean_im = temp.mean(axis=0)
            low_vals = temp < np.min(temp)*2
            temp[low_vals]=0
            entry[i] = [image_uncertainty(temp,crop=crop,uncertainty=uncertainty)/mean_im.max(),mean_im]
            del temp
    return entry

def get_stokes(processed,uncertainty=False,norm=False):
    Q = processed['H']-processed['V']
    U = processed['D']-processed['A']
    V = processed['R']-processed['L']
    I = np.sqrt(Q**2+U**2+V**2)
    if norm:
        I = I/1000
        Q = np.nan_to_num(Q/I)
        U = np.nan_to_num(U/I)
        V = np.nan_to_num(V/I)
        #I = np.nan_to_num(I/I)
    if uncertainty:
        nQ = unp.nominal_values(Q)
        nU = unp.nominal_values(U)
        nV = unp.nominal_values(V)
    else:
        nQ = Q
        nU = U
        nV = V
    vect = np.zeros((Q.shape[0],Q.shape[1],4))
    norm = np.zeros((Q.shape[0],Q.shape[1],4))

    for i,v in enumerate(vect):
        for j, v2 in enumerate(v):
            vect[i,j]=[I[i,j],Q[i,j],U[i,j],V[i,j]]
            norm[i,j]=[I[i,j],Q[i,j],U[i,j],V[i,j]]/I[i,j]
    norm = np.nan_to_num(norm)
    vect = np.nan_to_num(vect)

    return {'Q':Q,'U':U,'V':V,'nQ':nQ,'nU':nU,'nV':nV,'I':I,'vect':vect,'norm':norm}

def ellipse(S):
    #IP = np.sqrt(np.square(S['nU'])+np.square(S['nQ'])+np.square(S['nV']))
    IP = np.sqrt(np.square(S['nU'])+np.square(S['nQ'])+np.square(S['nV']))
    #theta = 0.5*np.arctan(np.true_divide(S['nU'], S['nQ'], where=(S['nU']!=0) | (S['nQ']!=0)))
    theta = 0.5*np.arctan2(S['nU'], S['nQ'])
    L = np.sqrt(np.square(S['nQ'])+np.square(S['nU']))
    A = np.sqrt(0.5*(IP+L))
    B = np.sqrt(0.5*(IP-L))
    h = np.sign(S['nV'])
    L = L/abs(L.max())
    B = B/abs(L.max())
    theta = theta*(180.0/np.pi)
    return [A,B,theta,h]

def plot_polarization(dataset,base_path=folder,resolution=1000,skip=1,crop=0,resize=False):
    sets = collect_data(base_path)
    processed = process_data(sets[dataset],crop=crop,resize=resize,base_path=base_path)
    stokes =  get_stokes(processed)
    ellipses = ellipse(stokes)

    grid = np.meshgrid(stokes['nQ'].shape[0],stokes['nQ'].shape[1])

    #plt.quiver(ellipses[0],ellipses[1],ellipses[3])
    #plt.show()
    img=stokes['I']
    orig_size = (img.shape[0]*10,img.shape[1]*10)
    img = imresize(img,orig_size)

    ells = []
    signs = []
    xy = 0
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    ax.imshow(img,origin='upper')

    for i, x in np.ndenumerate(stokes['nQ']):
        ells = Ellipse(xy=[i[0]*10,i[1]*10],width=ellipses[0][i],height=ellipses[1][i],angle=ellipses[2][i],
                        fill=False, ls='solid')
        signs = ellipses[3][i]
        xy = i
    
        if signs == -1:
            sign=np.array([1,0,0])
        if signs == 1:
            sign=np.array([0,0,1])
        if signs == 0:
            sign=np.array([0,0,0])
    
        ax.add_artist(ells)
        ells.set_edgecolor(sign)
        ells.set_facecolor(sign)
        ells.set_linewidth(0.3)

    ax.set_xlim(0,xy[0]*10)
    ax.set_ylim(0,xy[1]*10)
    plt.savefig(dataset+'.png', dpi=resolution)


class polarization():
    def __init__(self,folder,norm=False):
        self.folder=folder
        self.sets = collect_data(folder)
        self.keys = self.sets.keys()
        self.norm = norm
    def process(self,dataset,crop,resize,uncertainty=False,img_fit=False):
        self.images = process_data(self.sets[dataset],crop=crop,
                                    resize=resize,base_path=self.folder,
                                    uncertainty=uncertainty)
        self.fit = dict.fromkeys(self.images.keys(),[])
        if (dataset[0] == 'L') or (dataset[0] == 'H'):
            if dataset[0]=='L':
                mod = 'L'
                for i in self.images:
                    if i != 'V':    
                        temp = proc_img(self.images[i][0])
                        temp.fmode(mod,int(dataset[4]),int(dataset[5]),[150,0,0,50,0,1])
                        temp.recalc()
                        self.fit[i] = temp.mode_fit
                        self.images[i]=temp.centred_img()
                    else: 
                        temp = proc_img(self.images[i][0])
                        temp.fmode(mod,int(dataset[1]),int(dataset[2]),[150,0,0,50,0,1])
                        temp.recalc()
                        self.fit[i] = temp.mode_fit
                        self.images[i]=temp.centred_img()

            else:
                mod = 'H'
                for i in self.images:
                    if i != 'V':    
                        temp = proc_img(self.images[i][0])
                        temp.fmode(mod,int(dataset[1]),int(dataset[2]),[150,0,0,50,0,1])
                        temp.recalc()
                        self.fit[i] = temp.mode_fit
                        self.images[i]=temp.centred_img()
                    else: 
                        temp = proc_img(self.images[i][0])
                        temp.fmode(mod,int(dataset[4]),int(dataset[5]),[150,0,0,50,0,1])
                        temp.recalc()
                        self.fit[i] = temp.mode_fit
                        self.images[i]=temp.centred_img()

        self.stokes = get_stokes(self.images,norm=self.norm)
    def no_fit(self,dataset,uncertainty=False):
        self.images = process_data(self.sets[dataset],crop=False,
                            resize=False,base_path=self.folder,
                            uncertainty=False)
        self.theor = theory.theoretical([dataset[0],int(dataset[1]),int(dataset[2])],
                                    [dataset[3],int(dataset[4]),int(dataset[5])],
                                    self.images['V'][0].shape[0],stokes=True,norm=self.norm)
        for i in self.theor.k.keys():
            self.images[i] = self.images[i][0]
        self.stokes = get_stokes(self.images,norm=self.norm)
    def img_fit(self,dataset,filenames=False):
        self.images = process_data(self.sets[dataset],crop=False,
                            resize=False,base_path=self.folder,
                            uncertainty=False)
        if filenames:
            pass
        else:
            self.theor = theory.theoretical([dataset[0],int(dataset[1]),int(dataset[2])],
                                    [dataset[3],int(dataset[4]),int(dataset[5])],
                                    self.images['V'][0].shape[0],stokes=True,norm=self.norm)
        t_max=0
        for i in self.theor.k.keys():
            self.images[i] = fit_img(self.images[i][0],self.theor.k[i])[0]
            if self.images[i].max() > t_max:
                t_max=self.images[i].max()
        for i in self.images.keys():
            self.images[i]=self.images[i]/t_max
        self.stokes = get_stokes(self.images,norm=self.norm)
    def only_img(self,dataset):
        self.images = process_data(self.sets[dataset],crop=False,
                            resize=False,base_path=self.folder,
                            uncertainty=False)
        for i in self.images.keys():
            self.images[i] = self.images[i][0]
        self.stokes = get_stokes(self.images,norm=self.norm)
       

    def complex_fit(self,dataset,hor,ver,mult=[1,1]):
        self.images = process_data(self.sets[dataset],crop=False,
                            resize=False,base_path=self.folder,
                            uncertainty=False)

        self.theor = theory.theoretical(hor,
                                    ver,
                                    self.images['V'][0].shape[0],stokes=True,mult=mult,norm=self.norm)
        t_max=0
        for i in self.theor.k.keys():
            self.images[i] = fit_img(self.images[i][0],self.theor.k[i])[0]
            if self.images[i].max() > t_max:
                t_max=self.images[i].max()
        for i in self.images.keys():
            self.images[i]=self.images[i]/t_max
        self.stokes = get_stokes(self.images,norm=self.norm)


    def polarizer(self,steps):
        self.linear = np.zeros((steps+1,self.stokes['Q'].shape[0],self.stokes['Q'].shape[1]))
        self.rh = np.zeros((self.stokes['Q'].shape[0],self.stokes['Q'].shape[1]))
        self.lh = np.zeros((self.stokes['Q'].shape[0],self.stokes['Q'].shape[1]))
        step = 2*np.pi/steps
        for i in range(steps+1):
            for j,v in enumerate(self.stokes['vect']):
                for k,v2 in enumerate(v):
                    self.linear[i,j,k] = (muller(step*i).dot(self.stokes['vect'][j,k]))[0]
                    if i == 0:
                        self.rh[j,k] = circular(1).dot(self.stokes['vect'][j,k])[0]
                        self.lh[j,k] = circular(-1).dot(self.stokes['vect'][j,k])[0]
                    else:
                        pass
        #return self.stepped
       
                          
    def plot(self,mode='save',disp_img='default',
            show=True,title=False,skip=1,scale=1,
            resolution=1000,bg=False,the=False):
        
        if the:
            s = self.theor.stokes
        else:
            s = self.stokes

        ellipses = ellipse(s)
        cmap = LinearSegmentedColormap.from_list('name', [(0,'purple'), (.22,'green'),(.45,'yellow'),(.67,'orange'), (1,'pink')])
        norm = plt.Normalize(0, 180)
        
        grid = np.meshgrid(self.stokes['nQ'].shape[0],
                            self.stokes['nQ'].shape[1])
        if disp_img=='default':
            img = self.stokes['I']
        else:
            img=disp_img
        
        if bg:
            img= io.imread(bg)
    
        orig_size = (img.shape[0]*10,img.shape[1]*10)
        labels = imresize((ellipses[2]+90).astype(int),orig_size)
        img = imresize(img,orig_size)

        ells = []
        signs = []
        xy = 0
        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        #cmap = pylab.get_cmap('PRGn')
        #cgen = [cmap(i) for i in range(180)]
        #img=label2rgb(labels,img,colors=cgen,alpha=0.2,kind='overlay')
        ax.imshow(img,cmap='Greys_r')

        for i, x in np.ndenumerate(self.stokes['nQ']):
            if ((np.mod(i[0],skip))==0 and (np.mod(i[1],skip))==0):
                ells = Ellipse(xy=[i[0]*10,i[1]*10],width=ellipses[0][i]*scale,
                                height=ellipses[1][i]*scale,angle=ellipses[2][i],
                                fill=False, ls='solid')
                signs = ellipses[3][i]
                xy = i
            
                if signs == -1:
                    sign=np.array([1,0,0])
                if signs == 1:
                    sign=np.array([0,0,1])
                if signs == 0:
                    sign=np.array([0,0,0])
            
                ax.add_artist(ells)
                ells.set_edgecolor(sign)
                ells.set_facecolor(sign)
                ells.set_linewidth(0.5)
            else:
                pass

        ax.set_xlim(0,xy[0]*10)
        ax.set_ylim(0,xy[1]*10)
        if mode == 'save':
            if title:
                fig.savefig(title+'.png', dpi=resolution)
            else:    
                fig.savefig(dataset+'.png', dpi=resolution)
        if show:
            plt.show()
        plt.close('all')

    def plot_alt(self,mode='save',disp_img='default',
            show=True,title=False,skip=1,scale=1,
            resolution=1000,bg=False,the=False):
        
        if the:
            s = self.theor.stokes
        else:
            s = self.stokes

        self.ellipses = ellipse(s)
        grid = np.meshgrid(self.stokes['nQ'].shape[0],
                            self.stokes['nQ'].shape[1])
        if disp_img=='default':
            img = s['I']
        else:
            img=disp_img
        
        if bg:
            img= io.imread(bg)
        #cmaplist = pylab.get_cmap('PiYG',180)
        #cmaplist = [cmap(i) for i in range(cmap.N)]
        orig_size = (img.shape[0]*10,img.shape[1]*10)
        label_img = imresize(np.abs(self.ellipses[2]),orig_size,'nearest',mode='F')
        img = imresize(img,orig_size)
        #img = label2rgb(label_img,img,alpha=0.15) #,colors=cmaplist

        ells = []
        signs = []
        xy = 0
        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        im = ax.imshow(img,cmap='Greys_r')
        im1= ax.imshow(label_img,interpolation='nearest',cmap='viridis',alpha=0.2)

        for i, x in np.ndenumerate(self.stokes['nQ']):
            if ((np.mod(i[0],skip))==0 and (np.mod(i[1],skip))==0):
                ells = Ellipse(xy=[i[1]*10,i[0]*10],width=self.ellipses[0][i]*scale,
                                height=self.ellipses[1][i]*scale,angle=self.ellipses[2][i],
                                fill=False, ls='solid')
                signs = self.ellipses[3][i]
                xy = i
            
                if signs == -1:
                    sign=np.array([1,0,0])
                if signs == 1:
                    sign=np.array([0,0,1])
                if signs == 0:
                    sign=np.array([0,0,0])
            
                ax.add_artist(ells)
                ells.set_edgecolor(sign)
                ells.set_facecolor(sign)
                ells.set_linewidth(0.5)
            else:
                pass
        ax.set_xlim(0,xy[1]*10)
        ax.set_ylim(0,xy[0]*10)
        fig.colorbar(im1, orientation='vertical')

        if mode == 'save':
            if title:
                fig.savefig(title+'.png', dpi=resolution)
            else:    
                fig.savefig(dataset+'.png', dpi=resolution)
        if show:
            plt.show()
        plt.close('all')

crops = [250,550,480,780]

def cropimgs(filename,startx,lengthx,starty,lengthy,output='',save=False):
    out = output+filename.split('.')[0].split('\\')[-1]
    im = io.imread(filename)
    empty = np.zeros((im.shape[0],lengthx,lengthy))
    for i,v in enumerate(im):
        empty[i]=im[i,startx:startx+lengthx,starty:starty+lengthy]
    if save:
        np.save(out+'.npy',empty)
    return empty

def muller(theta):
    c = np.cos(2*theta)
    s = np.sin(2*theta)
    mat = 0.5*np.array([[1,c,s,0],
                    [c,c**2,s*c,0],
                    [s,s*c,s**2,0],
                    [0,0,0,0]])
    return mat

def circular(h):
        return  0.5*np.array([[1,0,0,h],
                    [0,0,0,0],
                    [0,0,0,0],
                    [h,0,0,1]])

def coherency(stokes):
    s0=stokes[0]
    s1=stokes[1]
    s2=stokes[2]
    s3=stokes[3]
    return 0.5*np.array(([s0+s1+0j,s2-s3*1j],[s2+s3*1j,s0-s1+0j]),dtype=complex)

def pol_coherency(stokes1,stokes2):
    return coherency(stokes1)+coherency(stokes2)

def deg_pol(coherence):
    return np.sqrt(1-( (4*np.linalg.det(coherence)) / (np.trace(coherence)**2)))

def radial_profile(data, center):
    y, x = np.indices((data.shape[0],data.shape[1]))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)
    r = r.ravel()
    data = data.reshape(-1,data.shape[-1])
    lis = [[] for i in range(r.max())]
    for i in range(len(lis)):
        for j in range(len(r)):
            if r[j] == i:

                lis[i].append(data[j])
            else:
                 pass
    return lis


def calc_all(v):
    v1 = np.rot90(v)
    v2 = np.rot90(v1)
    v3 = np.rot90(v2)
    temp = np.zeros((v.shape[0],v.shape[1],3),dtype=complex)
    coher = np.zeros((3,v.shape[0],v.shape[1],2,2),dtype=complex)
    final = np.zeros((v.shape[0],v.shape[1]))
    for i,val in enumerate(v):
        for j, val2 in enumerate(v):
            coher[0,i,j] = pol_coherency(v[i,j],v1[i,j])
            coher[1,i,j] = pol_coherency(v[i,j],v2[i,j])
            coher[2,i,j] = pol_coherency(v[i,j],v3[i,j])
            temp[i,j,0] = np.sqrt(8-4*deg_pol(pol_coherency(v[i,j],v1[i,j]))**2)
            temp[i,j,1] = np.sqrt(8-4*deg_pol(pol_coherency(v[i,j],v2[i,j]))**2)
            temp[i,j,2] = np.sqrt(8-4*deg_pol(pol_coherency(v[i,j],v3[i,j]))**2)
            final[i,j] = temp[i,j].max()
    coher = np.nan_to_num(coher)
    temp = np.nan_to_num(temp)
    final = np.nan_to_num(final)
    return temp, final, coher

def fidelity(measured,theory):
    fid = np.zeros((measured.shape[0],measured.shape[1]))
    for i,v in enumerate(measured):
        for j,v2 in enumerate(measured):
            th = sqrtm(theory[i,j],0)[0]
            mat_1 = np.matmul(measured[i,j],th)
            mat_2 = np.matmul(th,mat_1)
            fid[i,j] = np.abs(np.trace(sqrtm(mat_2,0)[0]))**2
    return fid


def createCircularMask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

        

def saveimg(base_path):
     f = [f for f in os.listdir(base_path)]
     for i in f:
        if i.split('.')[-1]=='npy':
            print i
            j=np.load(base_path+i).mean(axis=0)
            plt.imshow(j,cmap='Greys_r')
            plt.savefig(base_path+i.split('.')[0]+'img.png')
            plt.close('all')
#plot_polarization('H10H01',crop=crops)
    # H = import_image('H10H01_H')
    # V = import_image('H10H01_V')
    # R = import_image('H10H01_R')
    # L = import_image('H10H01_L')
    # A = import_image('H10H01_A')
    # D = import_image('H10H01_D')
# t=polarization('./stokes/')
# t.process(t.keys[0],False,3)
# t.polarizer(20)
# plt.close('all')
# plt.imshow(t.linear[0])
# plt.imshow(t.rh)
# plt.imshow(t.lh)
# plt.show()
