# - * - coding: utf-8 - * -
"""
Created on Mon Jun 01 09:53:19 2015

"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import gridspec
#from pandas import DataFrame as DF
from scipy.io import readsav
from astropy.io import fits

# longs = np.zeros(0, float)

# a   : semi-major axis of planetary orbit (in Rstar)
# inc : inclination angle of orbital plane (in degrees)
# per : period of planetary orbit (in days), assumed circular orbit
# r   : stellar radius (in pixel)
# rp  : planetary radius (in Rstar) (1 Rjup = 6.9911e4 km = 0.100447 Rsun)
# tef : effective surface temperature (in kelvin)
# u1  :
# u2  :
# star_image  : white light image of star (or Sun)
# x0  :

#K63  = readsav('paramk63.save')

## convert -delay 10 -loop 1 ./K63_transit/*.png K63_transit.gif
run_model = 1
create_gif = 1
samples = 10
## use_last = 1

 
# arquivo FITS dos dados
nome = 'aia_171_level1.fits'
file171 = 'aia_lev1_171a_2022_10_01t13_30_09_35z_image_lev1.fits'
file1700 = 'aia.lev1.1700A_2014-02-25T00_44_30.71Z.image_lev1.fits'

hdul = fits.open(file171)
star_image = hdul[0].data # OS DADOS (IMAGENS) ESTAO AQUI

a = 19.55 ## semi-eixo maior da orbita (quase certeza que a orbita é circular)
inc = 92.7 # 90-2.9 ## 90 graus é no equador (confirmar!)
per = 9.43415 ## periodo (nao precisar mudar)
tef = 5576.0

u1 = 0.59 # not used
u2 = 0.0 # not used
x0 = 177.843 # not used

rp = 0.0662 ## raio do planeta
### r = 368.65082

# r_pix = arcsec / (arcsec/pixel) = pixel
r = hdul[0].header['RSUN_OBS']/hdul[0].header['CDELT1'] # radius in arcsec


N = len(star_image[:, 0])

Rplan = rp * r      # planet radius in pixels
Rorbit = a * r      # orbital radius in pixels
alpha = np.radians(inc) # inclination angle of orbital plane [rad]


alpha_limit = np.arctan(r/Rorbit) + np.pi/2

if alpha < alpha_limit:
    dt = 30   
    t = per * 24. * dt
    
    phi = np.arange((t+0.5) - 1) * 360./(t)
    
    ii0 = np.where((180<=phi) & (phi<=360))
    phi = np.radians(phi[ii0])
    
    #theta = np.arccos(1.2*r/Rorbit)
    #phi = np.linspace(np.pi+theta,2*np.pi-theta,samples)
    
    Xorig = Rorbit * np.cos(phi)
    Yorig = Rorbit * np.sin(phi) * np.cos(alpha)
    
    edge = (N-2*r)/2/r+1 # 
    ii1 = np.where((abs(Xorig)<N/2*edge) & (abs(Yorig)<N/2))[0]
    X = Xorig[ii1];   Y = Yorig[ii1]
    TotalFlux = np.zeros(len(ii1))+np.sum(star_image)
    
    ii2 = np.argsort(X)
    X = X[ii2];   Y = Y[ii2]
    
    # show star and planet path
    image = star_image
    image[np.where(star_image<=0)]=1
    plt.imshow(np.log10(image),cmap='copper',aspect='equal',origin='lower')
    phi2 = np.linspace(0,2*np.pi,50)
    plt.plot(N/2+r*np.cos(phi2),N/2+r*np.sin(phi2),'--') # plot radius
    plt.plot(X+N/2,Y+N/2) # plot path
    
    Xstar, Ystar = X/r, Y/r
    
    X_wl = (np.arange(0, star_image.shape[0], 1) - star_image.shape[0]/2) / r
    Y_wl = (np.arange(0, star_image.shape[1], 1) - star_image.shape[1]/2) / r
    X_wl, Y_wl = np.meshgrid(X_wl, Y_wl)# X, Y matrix coordinates in Rstar
    
    RR_st = np.sqrt(Xstar**2 + Ystar**2)
    ii3 = np.where(RR_st==np.nanmin(RR_st))[0]
    lc_steps = (np.arange(len(ii1)) - ii3) / dt # light curve steps in hours
    
    # lat = (-np.arcsin((a) * np.cos(np.radians(inc))))/0.0174533
    # print(lat)
    # x1 = np.arange(0, N, dtype=float)-N/2
    # y1 = np.arange(0, N, dtype=float)-N/2
    
    nn = np.arange(N * N)
    
    ii4 = np.where((X+(N/2)>0) & (X+(N/2)<N) & (Y+(N/2)>0) & (Y+(N/2)<N))[0]
    i0 = ii4[0];   i1 = ii4[-1]
    
    
    # Model loop:
    if run_model:
        for i in np.arange(i0, i1):
          print(i)
          x0 = X[i]+N/2;   x0star = x0/r
          y0 = Y[i]+N/2;   y0star = y0/r
        
          planet = np.ones((1, N*N), dtype=float) # matriz de N por N
          ii5 = np.where(np.sqrt((nn/N-y0)**2 + (nn-N*np.floor(nn/N)-x0)**2)<Rplan)[0]
          planet[0][ii5] = 0
          planet = planet.reshape(-1, N)
        
          Eclipse = star_image * planet
          TotalFlux[i] = np.sum(Eclipse, dtype=float)
        
        light_curve = TotalFlux/TotalFlux[0]
        lc_min, lc_max = np.nanmin(light_curve), np.nanmax(light_curve)
        yl = (lc_max-lc_min)*.1;   ylim = lc_min-yl, lc_max+yl
        
        ## SHOW TRANSIT
        fig1=plt.figure()
        plt.plot(lc_steps,light_curve,'-x')
        
        fig2=plt.figure()
        plt.imshow(np.log10(image),cmap='gray',aspect='equal',origin='lower')
        plt.plot(X+N/2,np.interp(light_curve, (light_curve.min(), light_curve.max()), (0, 4000)))
        plt.plot(X+N/2,Y+N/2,'--r')
        
        np.save('lcsteps.npy',lc_steps)
        np.save('lc.npy',light_curve)
        np.save('X.npy',X)
        np.save('Y.npy',Y)
        np.save('image.npy',image)
        
    if create_gif:
        ax_col = '#707070';   cmap = 'copper'# 'gist_heat'#
        sav_dir = 'K63_transit/'
        if not os.path.isdir(sav_dir): os.makedirs(sav_dir)
        
        plot_range = samples/2 # np.arange(i0, i1, 2)
        print(i0, 'to', i1, 'len:', len(plot_range))
        for i in plot_range:
          x0 = X[i]+N/2;   x0star = x0/r
          y0 = Y[i]+N/2;   y0star = y0/r
        
          planet = np.ones((1, N*N), dtype=float) # matriz de N por N
          ii5 = np.where(np.sqrt((nn/N-y0)**2 + (nn-N*np.floor(nn/N)-x0)**2)<Rplan)[0]
          planet[0][ii5] = 0
          planet = planet.reshape(-1, N)
        
          Eclipse = star_image * planet
        
          # Plots:
          fig = plt.figure(figsize=(6, 6))
          gs = gridspec.GridSpec(10, 1)#, width_ratios=[1, 1])
        
          ax = plt.subplot(gs[:6, :]);   ax.set_aspect(1);   ax.set_facecolor('k')
          ax.pcolormesh(X_wl, Y_wl, Eclipse, cmap=cmap, shading='gouraud')
          ax.plot(Xstar, Ystar, '-.', lw=1, alpha=.7)
          if 0:# marcação do planeta
            xplan = np.mean(X_wl.ravel()[ii5]);   yplan = np.mean(Y_wl.ravel()[ii5])
            ax.plot(xplan, yplan, '.,')
          ax.grid(c=ax_col, alpha=.3, lw=.5)
          ax.set_xlabel('$R_{star}$')
          ax.set_ylabel('$R_{star}$')
          ax.set_xlim(Xstar[0], Xstar[-1])
          for sp in ('left','bottom','right','top'): ax.spines[sp].set_color(ax_col)
          ax.xaxis.label.set_color(ax_col);   ax.yaxis.label.set_color(ax_col)
          ax.tick_params(axis='both', which='both', colors=ax_col)
        
          ax = plt.subplot(gs[7:, :]);   ax.set_facecolor('k')
          ax.plot(lc_steps[i0:i], light_curve[i0:i])
          ax.grid(c=ax_col, alpha=.3, lw=.5)
          ax.set_xlabel('time (h)')
          ax.set_ylabel('relative flux')
          ax.set_xlim(lc_steps[i0], lc_steps[i1])
          ax.set_ylim(ylim)
          for sp in ('left','bottom','right','top'): ax.spines[sp].set_color(ax_col)
          ax.xaxis.label.set_color(ax_col);   ax.yaxis.label.set_color(ax_col)
          ax.tick_params(axis='both', which='both', colors=ax_col)
        
          fig.savefig(sav_dir+'K63_transit_%04d'%i, facecolor='#010101FF', dpi=80)
          fig.clf()
        
          print(i)
else:
    print('NO TRANSIT ++++++++++++++++++++++++++++++')

#%% #==========================================================================
# ORIGINAL CODE:
# =============================================================================

# # -*- coding: utf-8 -*-
# """
# Created on Mon Jun 01 09:53:19 2015

# """

# import numpy as np
# from scipy.io.idl import readsav
# from matplotlib.pyplot import *
# from matplotlib import pyplot
# from scipy import interpolate
# from scipy import ndimage
# import scipy
# import math
# import keyword
# import pandas as pd

# dt=20

# longs=np.zeros(0,float)

# #caminho= 'Dropbox/projeto_programacao'
# #s = readsav(caminho+'/'+'paramsave')
# s = readsav('paramsave')

# n=len(s.wl[:,0])

# Rs=s.rp*s.r
# ar=s.a*s.r
# alfa=math.radians(s.inc)

# teta=np.arange((s.per*24.*float(dt)+0.5)-1)*360./(s.per*24*float(dt))
# ii=np.where((teta >= 180) & (teta <= 360))
# teta=np.radians(teta[ii])
# xx=ar*np.cos(teta)
# yy=ar*np.sin(teta)*math.cos(alfa)

# xxp=xx
# yyp=yy

# pp=np.where((abs(xxp) < n/2*1.2) & (abs(yyp) < n/2))

# xp=xxp[pp]
# yp=yyp[pp]

# jj=np.argsort(xp)
# xp=xp[jj]
# yp=yp[jj]

# wp=np.zeros(len(pp[0]))+np.sum(s.wl)

# lat=(-math.asin((s.a)*math.cos(math.radians(s.inc))))/0.0174533
#    ##print (lat)
# x1=np.arange(0,n,dtype=float)-n/2
# y1=np.arange(0,n,dtype=float)-n/2

# kk=np.arange(n*n)
# jj=np.where((xp+(n/2) > 0) & (xp+(n/2) < n) & (yp+n/2 > 0) & (yp+(n/2) < n))
# i0=jj[0][0]
# i1=jj[0][len(jj[0])-1]

# for i in range(i0,i1):
#     plan=np.zeros((n,n),dtype=float)+1 ##matriz de n por n
#     x0=xp[i]+n/2
#     y0=yp[i]+n/2
#     ii=np.where((kk/n-y0)**2+(kk-n*np.floor(kk/n)-x0)**2 < Rs**2)
#     plan=np.reshape(plan,(1,np.product(plan.shape)))
#     plan[0][ii[0]]=0
#     np.shape(plan)
#     plan=np.reshape(plan,(-1,n))
#     ##plan=pd.rolling_mean(plan,3,center=True)
#     np.shape(plan)
#     wp[i]=np.sum(s.wl*plan,dtype=float)
#     pyplot.subplot(211)
#     imshow(np.copy(s.wl*plan),cmap="gray",origin='lower',extent=[x1[0],x1[len(x1)-1],y1[0],y1[len(y1)-1]])
#     pyplot.plot(xp,yp,'-' )
#     pyplot.axis([x1[0],x1[len(x1)-1],y1[0],y1[len(y1)-1]])
#     ang=np.arange((np.radians(181)*2))
#     lc=wp/wp[0]
#     #print(lc)
#     d=np.sqrt(xp**2+yp**2)
#     ii=np.where(d==min(d))
#     ts=np.arange(len(pp[0]))-ii[0]/dt
#     pyplot.subplot(212)
#     pyplot.plot(ts[i0:i],lc[i0:i])
#     pyplot.axis([ts[0],ts[len(ts)-1],min(lc),lc[0]+0.001])
#     pyplot.waitforbuttonpress(.1)
#    # pyplot.close()
#     pyplot.show()