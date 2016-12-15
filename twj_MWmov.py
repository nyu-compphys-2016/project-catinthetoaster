from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pdb as pdb
from math import pi
import pickle as pickle


munot=1
blue=(114/255,158/255,206/255)
orange=(255/255,158/255,74/255)
green=(103/255,191/255,92/255)
red=(237/255,102/255,93/255)
purple=(173/255,139/255,201/255)


def vecMHD(rho,vx,vy,vz,Bx,By,Bz,P):
    #rho,vx,vy,vz,Bx,By,Bz,P=np.copy(q[0,:]),np.copy(q[1,:]),np.copy(q[2,:]),np.copy(q[3,:]),np.copy(q[4,:]),np.copy(q[5,:]),np.copy(q[6,:]),np.copy(q[7,:])
    #pdb.set_trace()
    Bsq=Bx*Bx+By*By+Bz*Bz
    vsq=vx*vx+vy*vy+vz*vz
    E=P/(gam-1)+.5*rho*vsq+Bsq*.5
    Bdv=Bx*vx+By*vy+Bz*vz
    Pst=P+Bsq/2
    r=np.array([rho,rho*vx,rho*vy,rho*vz,E,Bx,By,Bz])
    F=np.array([rho*vx,rho*vx*vx+P+.5*Bsq-Bx*Bx,rho*vx*vy-Bx*By,rho*vx*vz-Bx*Bz,(E+Pst)*vx-Bdv*Bx,np.repeat(0,rho.size),By*vx-Bx*vy,Bz*vx-Bx*vz])
    return r, F

def compMHD(r):
    rho=np.copy(r[0,:])
    vx,vy,vz=r[1,:]/rho,r[2,:]/rho,r[3,:]/rho
    By=r[6,:]
    Bz=r[7,:]
    Bx=r[5,:]
    Bsq=Bx*Bx+By*By+Bz*Bz
    vsq=vx*vx+vy*vy+vz*vz
    P=(r[4,:]-.5*rho*vsq-Bsq*.5)*(gam-1)
    return np.array([rho,vx,vy,vz,Bx,By,Bz,P])

def minmod(x,y,z):
    return .25*np.abs(np.sign(x)+np.sign(y))*(np.sign(x)+np.sign(z))*np.amin([np.abs(x),np.abs(y),np.abs(z)],axis=0)

def cmethod(r):
    cr=np.zeros([r.shape[0],(N+1)*2])
    thet=1.5
    cr[:,::2]=r[:,1:N+2]+0.5*minmod(thet*(r[:,1:N+2]-r[:,:N+1]),0.5*(r[:,2:N+3]-r[:,:N+1]),thet*(r[:,2:N+3]-r[:,1:N+2]))
    cr[:,1::2]=r[:,2:N+3]-0.5*minmod(thet*(r[:,2:N+3]-r[:,1:N+2]),0.5*(r[:,3:N+4]-r[:,1:N+2]),thet*(r[:,3:N+4]-r[:,2:N+3]))
    return cr

def LU(r,F,delx,tmpt=None):
    cr=compMHD(r)
    vx=cr[1,:]
    Bx=cr[4,:]
    By=cr[5,:]
    Bz=cr[6,:]
    P=cr[7,:]
    bx=Bx/np.sqrt(cr[0,:])
    Csq=(gam*P/cr[0,:])
    #vsq=cr[1,:]*cr[1,:]+cr[2,:]*cr[2,:]+cr[3,:]*cr[3,:]
    bsq=(Bx*Bx+By*By+Bz*Bz)/cr[0,:]
    #H=(cr[4,:]+cr[6,:]+bsq*.5)/cr[0,:]
    #a=(gam-1)*(H-vsq*.5-bsq/cr[0,:])
    Ca=np.sqrt(Bx*Bx/(munot*cr[0,:])) # CHECK TO SEE IF ITS VX
    Cf=np.sqrt(.5*(Csq+bsq+np.sqrt((Csq+bsq)**2-4*Csq*bx*bx)))
    Cs=np.sqrt(.5*(Csq+bsq-np.sqrt((Csq+bsq)**2-4*Csq*bx*bx)))
    lambp=np.array([vx+Cs,vx+Ca,vx+Cf])
    lambm=np.array([vx-Cs,vx-Ca,vx-Cf])
    alphp=np.amax([np.repeat(0,N+1),lambp[0,::2],lambp[1,::2],lambp[2,::2],lambp[0,1::2],lambp[1,1::2],lambp[2,1::2]],axis=0)
    alphm=np.amax([np.repeat(0,N+1),-1*lambm[0,::2],-1*lambm[1,::2],-1*lambm[2,::2],-1*lambm[0,1::2],-1*lambm[1,1::2],-1*lambm[2,1::2]],axis=0)
    flag=0
    if tmpt == None:
        flag=1
        tmpt=delx*.5/max(np.append(alphp,alphm))
    Fip=((alphp[1:]*F[:,2::2]+alphm[1:]*F[:,3::2]-alphp[1:]*alphm[1:]*(r[:,3::2]-r[:,2::2]))/(alphp[1:]+alphm[1:]))
    Fim=((alphp[:-1]*F[:,:-2:2]+alphm[:-1]*F[:,1:-1:2]-alphp[:-1]*alphm[:-1]*(r[:,1:-1:2]-r[:,:-2:2]))/(alphp[:-1]+alphm[:-1]))
    #pdb.set_trace()
    if flag:
        return tmpt*(Fip-Fim)/delx, tmpt
    else:
        return tmpt*(Fip-Fim)/delx
    
def hydroHO(xgrid,tf,ttot,rho,vx,vy,vz,Bx,By,Bz,P):
    N=xgrid.size
    delx=(xgrid[-1]-xgrid[0])/N
    r,F=vecMHD(rho,vx,vy,vz,Bx,By,Bz,P)
    jq=np.array([rho,vx,vy,vz,Bx,By,Bz,P])
    if np.any(np.isnan(jq)):
        pdb.set_trace()
    r1=None
    r2=None
    jdelt=None
    for j in range(3):
        cq=cmethod(jq)
        jr,jF=vecMHD(*cq)
        if j == 0:
            Utmp, jdelt=LU(jr,jF,delx)
            if ttot+jdelt > tf:
                jdelt=tf-ttot
                Utmp=LU(jr,jF,delx,tmpt=jdelt)
        else:
            Utmp=LU(jr,jF,delx,tmpt=jdelt)
        if j == 0:
            r1=np.copy(r)
            r1[:,2:-2]=r[:,2:-2]-Utmp
            jq=compMHD(r1)
        if j == 1:
            r2=np.copy(r)
            r2[:,2:-2]=.75*r[:,2:-2]+.25*r1[:,2:-2]-.25*Utmp
            jq=compMHD(r2)
        if j == 2:
            r[:,2:-2]=(1/3)*r[:,2:-2]+(2/3)*r2[:,2:-2]-(2/3)*Utmp
    if np.any(np.isnan(r)):
        pdb.set_trace()
    q=compMHD(r)
    return q,jdelt







N=1000
ttot=0
vy=np.array(np.repeat(0.0,N+4))
vz=np.copy(vy)
Bx=np.repeat(0,N+4)
Bz=np.copy(Bx)
rhonot=1
Pnot=0.6
gam=5/3.0
alph=0.2
xnot=2
sig=0.4
xgrid=np.linspace(0,4,N)
tmp=np.append([-99,-99],xgrid)
tmp=np.append(tmp,[-99,-99])
fx=np.zeros(tmp.size)
cutx= np.abs(xgrid-xnot) < 1.0*sig
cut=np.abs(tmp-xnot) < 1.0*sig
fx[cut]=(1-((xgrid[cutx]-xnot)/sig)**2)**2
#rho=rhonot*(1+alph*fx)
rho=np.repeat(rhonot,N+4)
By=(1+alph*fx)
P=Pnot*(rho/rhonot)**gam
css=np.sqrt(gam*P/rho)
csso=np.sqrt(gam*Pnot/rhonot)
#vx=2/(gam-1)*(css-csso)
vx=np.copy(vy)


q=np.array([rho,vx,vy,vz,Bx,By,Bz,P])
tf=1
count=0
s=np.log(P[2:-2]*rho[2:-2]**-gam)
tarr=0
pltcount=0
while ttot < tf:
    q,delt=hydroHO(xgrid,tf,ttot,*q)
    ttot+=delt
    if not count%2:
        fig,ax=plt.subplots(4,1,figsize=(12,9))
        print ttot
        ax[0].plot(xgrid,q[0,2:-2],color=red,alpha=.5,lw=2,ls='--')
        ax[2].plot(xgrid,q[1,2:-2],color=red,alpha=.5,lw=2,ls='--')
        ax[1].plot(xgrid,q[7,2:-2],color=red,alpha=.5,lw=2,ls='--')
        ax[3].plot(xgrid,q[5,2:-2],color=red,alpha=.5,lw=2,ls='--')
        ax[0].plot(xgrid,rho[2:-2],label='Initial',color=blue,lw=2)
        ax[2].plot(xgrid,vx[2:-2],label='Initial',color=blue,lw=2)
        ax[1].plot(xgrid,P[2:-2],label='Initial',color=blue,lw=2)
        ax[3].plot(xgrid,By[2:-2],label='Initial',color=blue,lw=2)
        ax[0].set_ylabel('Density')
        ax[2].set_ylabel('Velocity')
        ax[1].set_ylabel('Pressure')
        ax[3].set_ylabel('By')
        ax[0].set_xlabel('Position')
        ax[1].set_xlabel('Position')
        ax[2].set_xlabel('Position')
        ax[3].set_xlabel('Position')
        tstr='t='+str(np.floor(ttot*100)/100)
        ax[0].plot(xgrid,q[0,2:-2],label=tstr,color=red,lw=2)
        ax[2].plot(xgrid,q[1,2:-2],label=tstr,color=red,lw=2)
        ax[1].plot(xgrid,q[7,2:-2],label=tstr,color=red,lw=2)
        ax[3].plot(xgrid,q[5,2:-2],label=tstr,color=red,lw=2)
        ax[0].set_ylabel('Density')
        ax[2].set_ylabel('Velocity')
        ax[1].set_ylabel('Pressure')
        ax[3].set_ylabel('B_y')
        ax[0].set_xlabel('Position')
        ax[1].set_xlabel('Position')
        ax[2].set_xlabel('Position')
        ax[3].set_xlabel('Position')
        ax[0].set_ylim(0.9,1.06)
        ax[1].set_ylim(0.50,0.66)
        ax[2].set_ylim(-0.08,0.08)
        ax[3].set_ylim(0.95,1.2)
        fig.suptitle('t= {0:.3g}'.format(ttot))
        ax[0].legend(loc=3)
        fig.savefig('plot_{0:03d}.png'.format(pltcount))
        pltcount+=1
        plt.close(fig)
#plt.savefig('f4.pdf')
    count+=1            

plt.show()

pdb.set_trace()

#1- 20 kpc
gam=2
ttot=0
#SET INITIAL CONDITIONS
N=1000
xgrid=np.linspace(3,20,N)
#Bx=np.append(np.repeat(0.75,N/2+2),np.repeat(0.75,N/2+2))
Bx=np.zeros(N+4)
cut=xgrid > 5
By=np.copy(Bx)
cutx=np.append([False,False],cut)
cutx=np.append(cutx,[False,False])
By[cutx]=5/xgrid[cut]
Bz=np.append([1,1],3/xgrid)
Bz=np.append(Bz,[3/xgrid[-1],3/xgrid[-1]])
vx=np.zeros(N+4)
vy=-1/np.sqrt(xgrid/20.0)+1/np.sqrt(2/20)
vy=np.append([vy[0],vy[0]],vy)
vy=np.append(vy,[vy[-1],vy[-1]])
vz=np.copy(vx)
rhonot=1
rho=rhonot*1/(1+(xgrid/3)**2)
rho=rho/rho.max()
rho=np.append([rho[0],rho[0]],rho)
rho=np.append(rho,[rho[-1],rho[-1]])
epsilon=1E-5
P=-1/np.sqrt(-1*xgrid+20+epsilon)+1
P[P < 0]=0
P=P/P.max()
P=np.append([P[0],P[0]],P)
P=np.append(P,[P[-1],P[-1]])
# thin=(-(xgrid-4)**2+16)
# thin[thin < 0]=0
# thin=10**thin
# thin=thin/thin.max()
# thick=


q=np.array([rho,vx,vy,vz,Bx,By,Bz,P])
tf=1
count=0
print 'wat'
while ttot < tf:
    q,delt=hydroHO(xgrid,tf,ttot,*q)
    ttot+=delt
    count+=1
    if not count%10:
        print ttot
        fig,ax=plt.subplots(4,1)
        
        ax[0].plot(xgrid,rho[2:-2],label='Initial',color=blue)
        ax[2].plot(xgrid,vx[2:-2],label='Initial',color=blue)
        ax[1].plot(xgrid,P[2:-2],label='Initial',color=blue) 
        ax[3].plot(xgrid,By[2:-2],label='Initial',color=blue)
        ax[0].set_ylabel('Density')
        ax[2].set_ylabel('Velocity')
        ax[1].set_ylabel('Pressure')
        ax[3].set_ylabel('By')
        ax[0].set_xlabel('Position')
        ax[1].set_xlabel('Position')
        ax[2].set_xlabel('Position')
        ax[3].set_xlabel('Position')
        tstr='t='+str(np.floor(ttot*100)/100)
        ax[0].plot(xgrid,q[0,2:-2],label=tstr,color=red)
        ax[2].plot(xgrid,q[1,2:-2],label=tstr,color=red)
        ax[1].plot(xgrid,q[7,2:-2],label=tstr,color=red)
        ax[3].plot(xgrid,q[5,2:-2],label=tstr,color=red)
        ax[0].set_ylabel('Density')
        ax[2].set_ylabel('Velocity')
        ax[1].set_ylabel('Pressure')
        ax[3].set_ylabel('B_y')
        ax[0].set_xlabel('Position')
        ax[1].set_xlabel('Position')
        ax[2].set_xlabel('Position')
        ax[3].set_xlabel('Position')
        plt.show()


