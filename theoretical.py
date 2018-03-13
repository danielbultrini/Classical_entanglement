import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from opencavity.beams import LgBasis
from opencavity.beams import HgBasis
from scipy.misc import imresize
from skimage import io
import matplotlib.animation as animation



 

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

class theoretical():
    def __init__(self,hor_beam,ver_beam,resolution=100,stokes=False,bas_size=130,mult=[1,1],norm=False):
        x=np.linspace(-300,300,resolution);y=x;x,y = np.meshgrid(x,y)
        self.grid = (x,y)
        self.norm=norm
        resolution2=resolution
        LG=LgBasis(1,bas_size)
        LG2=LgBasis(1,bas_size)
        LGV=LgBasis(1,bas_size)
        LGV2=LgBasis(1,bas_size)
        HG=HgBasis(1,bas_size,bas_size)
        HG2=HgBasis(1,bas_size,bas_size)
        HGV=HgBasis(1,bas_size,bas_size)
        HGV2=HgBasis(1,bas_size,bas_size)
        z=0.000000000000000001
        Image=False
        if hor_beam[0]=='H':
            Hor = (HG.generate_hg(hor_beam[1],hor_beam[2],x,y,z))
        elif hor_beam[0]=='HH':
            Hor = (HG.generate_hg(hor_beam[1],hor_beam[2],x,y,z))*hor_beam[5]+(HG2.generate_hg(hor_beam[3],hor_beam[4],x,y,z))*hor_beam[6]
        elif hor_beam[0]=='L':
            Hor = (LG.generate_lg(hor_beam[2],hor_beam[1],x,y,z))
        elif hor_beam[0]=='LL':
            Hor = (LG.generate_lg(hor_beam[2],hor_beam[1],x,y,z))*hor_beam[5]+(LG2.generate_lg(hor_beam[4],hor_beam[3],x,y,z))*hor_beam[6]
        else:
            Hor = np.array(io.imread(hor_beam[0],as_grey=True),dtype=np.complex128)*hor_beam[1]
            Image=True
        Hor = Hor*mult[0]



        if ver_beam[0]=='H':
            Ver = (HGV.generate_hg(ver_beam[1],ver_beam[2],x,y,z)*1j)
        elif ver_beam[0]=='HH':
            Ver = (HGV.generate_hg(ver_beam[1],ver_beam[2],x,y,z))*ver_beam[5]+(HGV2.generate_hg(ver_beam[3],ver_beam[4],x,y,z))*ver_beam[6]
        elif ver_beam[0]=='LL':
            Ver = (LGV.generate_lg(ver_beam[2],ver_beam[1],x,y,z))*ver_beam[5]+(LGV2.generate_lg(ver_beam[4],ver_beam[3],x,y,z))*ver_beam[6]

        elif ver_beam[0]=='L':
            Ver = (LGV.generate_lg(ver_beam[2],ver_beam[1],x,y,z)*1j)
        else:
            Ver = np.array(io.imread(ver_beam[0],as_grey=True),dtype=np.complex128)*ver_beam[1]
            resolution, resolution2= Ver.shape[0] , Ver.shape[1]
            Image=True
        Ver = Ver*mult[1]
        self.Ver=Ver
        self.Hor=Hor
        intensity = (np.conj(Ver)*Ver+np.conj(Hor)*Hor).astype(float)
        self.Jones = np.zeros((resolution,resolution2,2),dtype=complex)

        for i,v in enumerate(self.Jones):
            for j,v2 in enumerate(v):
                self.Jones[i,j,0]=Hor[i,j]
                self.Jones[i,j,1]=Ver[i,j]

        Horizontal = np.array(([1,0],[0,0]),dtype=complex)
        Vertical = np.array(([0,0],[0,1]),dtype=complex)
        Diag = np.array(([0.5,0.5],[0.5,0.5]),dtype=complex) 
        Adiag = np.array(([0.5,-0.5],[-0.5,0.5]),dtype=complex) 
        RH = np.array(([0.5,0.5j],[-0.5j,0.5]),dtype=complex) 
        LH = np.array(([0.5,-0.5j],[0.5j,0.5]),dtype=complex) 

        polarizers = [Horizontal,Vertical,Diag,Adiag,RH,LH]

        H = np.zeros((resolution,resolution2,2),dtype=complex)
        V = np.zeros((resolution,resolution2,2),dtype=complex)
        R = np.zeros((resolution,resolution2,2),dtype=complex)
        L = np.zeros((resolution,resolution2,2),dtype=complex)
        A = np.zeros((resolution,resolution2,2),dtype=complex)
        D = np.zeros((resolution,resolution2,2),dtype=complex)

        self.arrays = [H,V,D,A,R,L]

        for i,v in enumerate(self.Jones):
            for j,v2 in enumerate(v):
                for k, pol in enumerate(polarizers):
                    self.arrays[k][i,j] = np.dot(polarizers[k],self.Jones[i,j])

        H2 = np.zeros((resolution,resolution2))
        V2 = np.zeros((resolution,resolution2))
        R2 = np.zeros((resolution,resolution2))
        L2 = np.zeros((resolution,resolution2))
        A2 = np.zeros((resolution,resolution2))
        D2 = np.zeros((resolution,resolution2))
        Intarrays = [H2,V2,D2,A2,R2,L2]

        for i, v in enumerate(self.arrays):
            for j, v2 in enumerate(v):
                for k, v3 in enumerate(v2):
                    if Image:
                        Intarrays[i][j,k]=np.sqrt(np.abs(v3[0])**2+np.abs(v3[1])**2)
                    else:
                        Intarrays[i][j,k]=np.dot(np.conj(v3),v3)


        self.k = {'A':Intarrays[3],'D':Intarrays[2],'H':Intarrays[0],'L':Intarrays[5],'R':Intarrays[4],'V':Intarrays[1]}
        if stokes:
            self.stokes = get_stokes(self.k,norm=self.norm)
            self.stokes['I'] = intensity
#t = polarization('./stokes/')
#t.stokes = stokes
#t.plot(title='LG',skip=2,scale=80,show=False)



def jones_linear(input,angle):
    c=np.cos(angle)
    s = np.sin(angle)
    pol = np.array(([c**2,s*c],[s*c,s**2]),dtype=complex)
    out = np.zeros(input.shape,dtype=complex)
    for i, v in enumerate(input):
        for j,v2 in enumerate(v):
            out[i,j]=np.dot(pol,input[i,j])
    return out

def jones_QWP(input, angle):
    c=np.cos(angle)
    s = np.sin(angle)
    qwp = np.exp(-1j*np.pi/4)*np.array(([c**2+1j*s**2,(1-1j)*s*c],[(1-1j)*s*c,s**2+1j*c**2]),dtype=complex)
    out = np.zeros(input.shape,dtype=complex)
    for i, v in enumerate(input):
        for j,v2 in enumerate(v):
            out[i,j]=np.dot(qwp,input[i,j])
    return out

def jones_int(input):
    intensity = np.zeros((input.shape[0],input.shape[1]))
    for i,v in enumerate(input):
        for j,v2 in enumerate(v):
            intensity[i,j] = np.abs(v2[0]+v2[1])
    return intensity

def C_measure(input):
    ang = [0,np.pi/8,np.pi/4,3*np.pi/8]
    qwp_0 = jones_QWP(input,0)
    qwp_22 = jones_QWP(input,np.pi/8)
    qwp_45 = jones_QWP(input,np.pi/4)
    qwp_66 = jones_QWP(input,3*np.pi/8)
    pol0,pol22,pol45,pol66 = np.zeros(4),np.zeros(4),np.zeros(4),np.zeros(4)
    qwp,pol = [qwp_0,qwp_22,qwp_45,qwp_66],[pol0,pol22,pol45,pol66]
    for i,v in enumerate(qwp):
        for j,v2 in enumerate(ang):
            pol[i][j]=jones_int(jones_linear(v,v2))[38,40]
    return pol

def coherence(I1,I2,I3,I4):
    return (I1+I2-I3-I4)/(I1+I2+I3-I4)

def bell(pol):
    tt=coherence(pol[0][0],pol[2][2],pol[2][0],pol[0][2])
    tt_=coherence(pol[0][1],pol[2][3],pol[2][1],pol[0][3])
    t_t=coherence(pol[1][0],pol[3][2],pol[2][0],pol[1][2])
    t_t_=coherence(pol[1][1],pol[3][3],pol[2][1],pol[1][3])
    return tt-tt_+t_t+t_t_

def linear(input,steps,func,loc=[38,38,40,40]):
    ang = np.linspace(0,2*np.pi,steps)
    pol = np.zeros(steps)
    for j,v2 in enumerate(ang):
        pol[j]=jones_int(func(input,v2))[loc[0]:loc[1],loc[2]:loc[3]].mean()
    return pol

def linear_img(input,steps,func):
    ang = np.linspace(0,2*np.pi,steps)
    pol = [None]*steps
    for j,v2 in enumerate(ang):
        pol[j]=jones_int(func(input,v2))
    return pol

def HWP(input,angle):
    c=np.cos(2*angle)
    s = np.sin(2*angle)
    hwp = np.exp(-1j*np.pi/2)*np.array(([c,s],[s,-1*c]),dtype=complex)
    out = np.zeros(input.shape,dtype=complex)
    for i, v in enumerate(input):
        for j,v2 in enumerate(v):
            out[i,j]=np.dot(hwp,input[i,j])
    return out
