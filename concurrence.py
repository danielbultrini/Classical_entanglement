import os 
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
from scipy import optimize
from matplotlib.colors import LinearSegmentedColormap



def collect_data(base_path):
    f = [f for f in os.listdir(base_path)]
    sets = {}
    for i in f:
        parts = i.split('.')[0]
        parts = parts.split('\\')[-1]
        parts = parts.split('_')
        if not parts[0] in sets:
            hol={}
            for j in f:
                parts2 = j.split('.')[0]
                parts2 = parts2.split('\\')[-1]
                parts2 = parts2.split('_')
                if not parts2[1] in hol:
                    hwp = {}
                    for k in f:
                        parts3 = k.split('.')[0]
                        parts3 = parts3.split('\\')[-1]
                        parts3 = parts3.split('_')
                        if not parts3[2] in hwp:
                            temp = None
                            hwp[parts3[2]] = temp
                    hol[parts2[1]]=hwp
            sets[parts[0]]=hol
    for i in f:
        parts = i.split('.')[0]
        parts = parts.split('\\')[-1]
        parts = parts.split('_')
        temp = np.load(base_path+i).mean(axis=0)
        temp = temp-temp.min()
        temp = temp/temp.max()
        temp = np.nan_to_num(temp)
        sets[parts[0]][parts[1]][parts[2]]=temp


    return sets

def complete_data(superset):
    for i in superset.keys():
        temp, temp2 = superset[i]['0']['22'], superset[i]['0']['67']
        superset[i]['0']['22'], superset[i]['0']['67'] = temp2, temp
        superset[i]['22']={}
        superset[i]['45']={}
        superset[i]['67']={}
        for j in superset[i]['0'].keys():
            superset[i]['22'][j]=rotate(superset[i]['0'][j],np.rad2deg(np.pi/4))
            superset[i]['45'][j]=rotate(superset[i]['0'][j],np.rad2deg(np.pi/2))
            superset[i]['67'][j]=rotate(superset[i]['0'][j],np.rad2deg(6*np.pi/8))
    return superset
            

def bells(dataset,cord=False):
    _0_0=dataset['0']['0']
    _0_2=dataset['0']['22']
    _0_4=dataset['0']['45']
    _0_6=dataset['0']['67']
    _2_0=dataset['22']['0']
    _2_2=dataset['22']['22']
    _2_4=dataset['22']['45']
    _2_6=dataset['22']['67']
    _4_0=dataset['45']['0']
    _4_2=dataset['45']['22']
    _4_4=dataset['45']['45']
    _4_6=dataset['45']['67']
    _6_0=dataset['67']['0']
    _6_2=dataset['67']['22']
    _6_4=dataset['67']['45']
    _6_6=dataset['67']['67']

    c1 = coherence(_0_0,_4_4,_4_0,_0_4,cord)
    c2 = coherence(_0_2,_4_6,_4_2,_0_6,cord)
    c3 = coherence(_2_0,_6_4,_6_0,_2_4,cord)
    c4 = coherence(_2_2,_6_6,_6_2,_2_6,cord)
    return np.abs(c1-c2+c3+c4)

def coherence(dh,d2h2,d2h,dh2,cord=False):
    
    if cord:
        dh=dh[cord[0]:cord[1],cord[2]:cord[3]].mean()
        d2h=d2h[cord[0]:cord[1],cord[2]:cord[3]].mean()
        dh2=dh2[cord[0]:cord[1],cord[2]:cord[3]].mean()
        d2h2=d2h2[cord[0]:cord[1],cord[2]:cord[3]].mean()
    C = (dh+d2h2-d2h-dh2)/(dh+d2h2+d2h+dh2)
    C = np.nan_to_num(C)
    return C



def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def plot_figures(figures, nrows = 1, ncols=1,fact=True):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """
    if fact:
        cmap = LinearSegmentedColormap.from_list('name', [(0,'black'), (1/np.sqrt(2),'white'), (1,'red')])
        norm = plt.Normalize(0, 2*np.sqrt(2))
    else:
        cmap = 'Greys'
        norm = None
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in zip(range(len(figures)), figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=cmap, norm=norm, interpolation='none')
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional


# tst1 = collect_data('C:/Users/danie/Documents/Final year/project/programs/extras/concorrect/')
# tst_temp = collect_data('C:/Users/danie/Documents/Final year/project/programs/extras/concorrect/')

# tst1_mach = complete_data(tst_temp)
# tst2 = collect_data('C:/Users/danie/Documents/Final year/project/programs/extras/images/')
# tst2 = complete_data(tst2)