import numpy as np

from math import log, trunc, sqrt
from skimage.io import imread, imsave
from scipy.misc import imresize
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps 
from scipy.special import ndtr
from scipy.stats import norm, chi2 as Chi2
from scipy.linalg import hadamard, sqrtm



def Hadamard2Walsh(n):
    # Function computes both Hadamard and Walsh Matrices of n=2^M order

    hadamardMatrix=hadamard(n)
    HadIdx = np.arange(n)
    M = int(log(n,2)+1)

    for i in HadIdx:
        s=format(i, '#032b')
        s=s[::-1]
        s=s[:-2]
        s=list(s)
        x=[int(x) for x in s]
        x=np.array(x)
        if(i==0):
            binHadIdx=x
        else:
            binHadIdx=np.vstack((binHadIdx,x))

    binSeqIdx = np.zeros((n,M)).T

    for k in reversed(range(1,int(M))):
        tmp=np.bitwise_xor(binHadIdx.T[k],binHadIdx.T[k-1])
        binSeqIdx[k]=tmp

    tmp=np.power(2,np.arange(M)[::-1])
    tmp=tmp.T
    SeqIdx = np.dot(binSeqIdx.T,tmp)

    j=1
    for i in SeqIdx:
        i = int(i)
        if(j==1):
            walshMatrix=hadamardMatrix[i]
        else:
            walshMatrix=np.vstack((walshMatrix,hadamardMatrix[i]))
        j+=1

    return (hadamardMatrix,walshMatrix)

def SaveFigureAsImage(fileName,fig=None,**kwargs):
    ''' Save a Matplotlib figure as an image without borders or frames.
       Args:
            fileName (str): String that ends in .png etc.

            fig (Matplotlib figure instance): figure you want to save as the image
        Keyword Args:
            orig_size (tuple): width, height of the original image used to maintain 
            aspect ratio.
    '''
    fig_size = fig.get_size_inches()
    w,h = fig_size[0], fig_size[1]
    fig.patch.set_alpha(0)
    if kwargs.has_key('orig_size'): # Aspect ratio scaling if required
        w,h = kwargs['orig_size']
        w2,h2 = fig_size[0],fig_size[1]
        fig.set_size_inches([(w2/w)*w,(w2/w)*h])
        fig.set_dpi((w2/w)*fig.get_dpi())
    a=fig.gca()
    a.set_frame_on(False)
    a.set_xticks([]); a.set_yticks([])
    plt.axis('off')
    plt.xlim(0,h); plt.ylim(w,0)
    fig.savefig(fileName, transparent=True, bbox_inches='tight', \
                        pad_inches=0)
def save_image(data, cm, fn):
   
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])
     
    fig = plt.figure()
    fig.set_size_inches(width/height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
 
    ax.imshow(data, cmap=cm, interpolation='none')
    plt.savefig(fn, dpi = height*3, format='png') 
    plt.close()

def generate_basis(n,ordered = 1, filepath = './full/',scale=1):
    H = Hadamard2Walsh(n)[ordered]
    H_T = H.transpose()
    base_name = 'NH_'+str(n)+'_'
    for i in xrange(len(H[0])):
        for j in xrange(len(H[0])):
            basis = np.outer(H_T[j],H[i])
            basis = imresize(basis,scale*100,interp='nearest')

            file_name = filepath+base_name+str(i)+'_'+str(j)+'.png'
            save_image(basis,'Greys',file_name)

def gen_basis(n,ordered = 1, filepath = './Nbase/',scale=1):
    H = Hadamard2Walsh(n)[ordered]
    H_T = H.transpose()
    x = 0
    for i in xrange(len(H[0])):
        for j in xrange(len(H[0])):
            basis = ((np.outer(H_T[j],H[i])+1)/2)
            basis.astype(int)
            file_name = str(x).format()+'.bmp'
            #img = Image.new('1',(128,128))
            basis = imresize(basis,scale*100,interp='nearest')
            img = Image.fromarray(basis)
            img = img.convert('1')

            img.save(filepath+str(x)+'.bmp')
            x+=1

def np_basis(n,ordered = 1,scale=1,resize=False):
    H = Hadamard2Walsh(n)[ordered]
    H_T = H.transpose()
    count = 0
    x = [np.empty((n,n,3),dtype=int)]*(n*n)
    for i in xrange(len(H[0])):
        for j in xrange(len(H[0])):
            basis = ((np.outer(H_T[j],H[i])+1)/2)
            basis = basis*255
            basis = np.uint8(basis)
            if scale !=1:
                basis = imresize(basis,scale*100,interp='nearest')
            if resize:
                basis = imresize(basis,resize,interp='nearest')
            x[count]=basis
            count +=1
    return x

def encode_img(n, filepath = 'image.png', ordered = 1):
    import scipy.ndimage as image
    img = image.imread(filepath, mode='L')    
    H = Hadamard2Walsh(n)[ordered]
    H_T = H.transpose()
    seq = np.zeros(n**2,dtype=int)
    x = 0
    for i in xrange(len(H[0])):
        for j in xrange(len(H[0])):
            basis = ((np.outer(H_T[j],H[i])+1)/2)
            basis = np.multiply(basis,img)
            seq[x]= basis.sum()
            #seq[x] = np.inner(basis,img)

            x+=1
    return seq

def decode_img(sequence,output='out.png',ordered=1,n=False,resize=False,scale=1):
    if n == False:
        n = int(np.sqrt(len(sequence)))
    H = Hadamard2Walsh(n)[ordered]
    H_T = H.transpose()
    out = np.zeros((n,n),dtype=float)
    if resize:
        out = np.zeros(resize,dtype=float)
    x = 0
    for i in xrange(len(H[0])):
        for j in xrange(len(H[0])):
            if x < len(sequence):
                basis = ((np.outer(H_T[j],H[i])+1)/2)
                if scale !=1:
                    basis = imresize(basis,scale*100,interp='nearest')
                if resize:
                    basis = imresize(basis,resize,interp='nearest')
                if x > 0: 
                    out += basis*(sequence[x])
                x+=1
    out = (out-out.min())/(n*n/2)

    return out

def diff_encode(img,n,ordered=1):
    H = Hadamard2Walsh(n)[ordered]
    H_T = H.transpose()
    seq = np.zeros(n**2,dtype=int)
    x = 0
    for i in xrange(len(H[0])):
        for j in xrange(len(H[0])):
            basis = ((np.outer(H_T[j],H[i])+1)/2)
            neg   = ((np.outer(H_T[j],H[i])*(-1)+1)/2)
            neg = np.multiply(neg,img)
            basis = np.multiply(basis,img)
            seq[x]= basis.sum()-neg.sum()
            x+=1
    return seq

def diff_decode(seq,n,ordered=1):
    pass 

def compressed_basis(n,ordered = 1, filepath = './blp/',scale=1):
    H = Hadamard2Walsh(n)[ordered]
    H_T = H.transpose()
    x = 0
    k=0
    m=0
    color= np.zeros((3,n,n),dtype='int32')
    base_name = 'NH_'+str(n)+'_'
    for i in range(len(H[0])):
        for j in range(len(H[0])):
            basis = ((np.outer(H_T[j],H[i])+1)/2)
            basis.astype('int32')
            if m < 2:
                if k < 7:
                    color[m]+=basis*(2**k)
                    k+=1
                elif k == 7:
                    color[m]+=basis*(2**k)
                    k=0
                    m+=1
            elif m == 2:
                if k < 7:
                    color[m]+=basis*(2**k)
                    k+=1
                elif k == 7:
                    color[m]+=basis*(2**k)
                    k=0
                    m+=1
            elif m == 3:
                m=0
                RGB = np.zeros((n,n,3))
                for w in range(n):
                    for h in range(n):
                        RGB[w,h,2] = color[0,w,h]
                        RGB[w,h,0] = color[1,w,h]
                        RGB[w,h,1] = color[2,w,h]
                #RGB=np.ascontiguousarray(RGB.transpose(1,2,0))
                if scale != 1:
                    RGB=imresize(RGB,scale*100,interp='nearest')
                RGB = RGB.astype('uint8')
                #np.save( filepath+str(x)+'.npy', RGB)
                #img = Image.fromarray(RGB,'RGB')
                #img.save(filepath+str(x)+'.bmp')
                imsave(filepath+str(x)+'.bmp', RGB, )
                x+=1
                color= np.zeros((3,n,n))






def WHT(x):
    x=np.array(x)
    if(len(x.shape)<2): # make sure x is 1D array
        if(len(x)>3):   # accept x of min length of 4 elements (M=2)
            # check length of signal, adjust to 2**m
            n=len(x)
            M=trunc(log(n,2))
            x=x[0:2**M]
            h2=np.array([[1,1],[1,-1]])
            for i in xrange(M-1):
                if(i==0):
                    H=np.kron(h2,h2)
                else:
                    H=np.kron(H,h2)

            return (np.dot(H,x)/2.**M, x, M)
        else:
            print("HWT(x): Array too short!")
            raise SystemExit
    else:
        print("HWT(x): 1D array expected!")
        raise SystemExit


def ret2bin(x):
    # Function converts list/np.ndarray values into +/-1 signal
    Y=[]; ok=False
    if('numpy' in str(type(x)) and 'ndarray' in str(type(x))):
        x=x.tolist()
        ok=True
    elif('list' in str(type(x))):
        ok=True
    if(ok):
        for y in x:
            if(y<0):
                Y.append(-1)
            else:
                Y.append(1)
        return Y
    else:
        print("Error: neither 1D list nor 1D NumPy ndarray")
        raise SystemExit

def bit_reverse_traverse(a):
    n = a.shape[0]
    assert(not n&(n-1) ) # assert that n is a power of 2
    if n == 1:
        yield a[0]
    else:
        even_index = np.arange(n/2)*2
        odd_index = np.arange(n/2)*2 + 1
        for even in bit_reverse_traverse(a[even_index]):
            yield even
        for odd in bit_reverse_traverse(a[odd_index]):
            yield odd

def get_bit_reversed_list(l):
    n = len(l)
    indexs = np.arange(n)
    b = []
    for i in bit_reverse_traverse(indexs):
        b.append(l[i])
    return b

def FWHT(X):
    x=get_bit_reversed_list(X)
    x=np.array(x)
    N=len(X)

    for i in range(0,N,2):
        x[i]=x[i]+x[i+1]
        x[i+1]=x[i]-2*x[i+1]

    L=1
    y=np.zeros_like(x)
    for n in range(2,int(log(N,2))+1):
        M=2**L
        J=0; K=0
        while(K<N):
            for j in range(J,J+M,2):
                y[K]   = x[j]   + x[j+M]
                y[K+1] = x[j]   - x[j+M]
                y[K+2] = x[j+1] + x[j+1+M]
                y[K+3] = x[j+1] - x[j+1+M]
                K=K+4
            J=J+2*M
        x=y.copy()
        L=L+1

    return x/float(N)

