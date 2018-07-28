import numpy as np
import math  as ma

from scipy.spatial import distance_matrix

# === Parameter
NA  =2
NXYZ=3
R0=1.25

#RMIN=-0.3
#RMAX= 0.3
RMIN=-5.3
RMAX= 5.3
DR  =0.2
NG  =int((RMAX-RMIN)/DR)

# Fitting parameter for 1s orbital, namely, sqrt(1/pi)*exp(-r)
CC=[0.444635,0.535328,0.154329]
OC=[0.109818,0.109818,0.405771]

# === Function
def setInitCoord(na=NA,nxyz=NXYZ): 
   rn=np.zeros((na,nxyz))
   for i in range(NA): 
      if i==0: 
         c=R0*(-1.)
      else: 
         c=R0
      rn[i,2]=c

   return rn

def setNucParameterForBasis(ng=NG,nxyz=NXYZ,rmin=RMIN,rmax=RMAX): 
   rz=np.linspace(rmin,rmax,ng)
   rb=np.zeros((ng,nxyz))
   rb[:,-1]=rz

   return rb

def calcDistance(r_mat): 
   return distance_matrix(r_mat,r_mat)

def calcS(r): 
   return np.exp(-r)*(1.+r+r*r/3.)

def calcT(r): 
   return np.exp(-r)*(1.+r-r*r/3.)*0.5

# --> cython
def calcVen(r,rn): 
   na=rn.shape[0]
   nb= r.shape[0]

   v=np.zeros((nb,nb))
   for i in range(nb): 
      for j in range(nb): 
         for ia in range(na):
            v[i,j]+=calcVGTO(rn[ia],r[i],r[j])

   return v

def calcVGTO(rc,ra,rb): 
   cc=np.array(CC)
   oc=np.array(OC)
   ngau=cc.shape[0]
   rab=np.linalg.norm(ra-rb)

   v=0.
   for i in range(ngau):
      ca=cc[i]*np.power(oc[i]/ma.pi*2.,0.75)
      for j in range(ngau):
         cb =cc[j]*np.power(oc[j]/ma.pi*2.,0.75)
         rp =(oc[i]*ra+oc[j]*rb)/(oc[i]+oc[j])
         rpc=np.linalg.norm(rp-rc)
         v +=ca*cb*VInt(oc[i],oc[j],rab,rpc)

   return v

def VInt(a,b,rab,rpc): 
   return -2.*ma.pi/(a+b)*np.exp(-a*b/(a+b)*rab*rab)*F0((a+b)*rpc*rpc)  

def F0(t): 
   if t<=0.000001: 
      f0=1.-t/3.
   else: 
      f0=0.5*ma.sqrt(ma.pi/t)*ma.erf(ma.sqrt(t))

   return f0

def calcX(s,u): 
   nb=s.shape[0]
   x =np.zeros((nb,nb))
   for i in range(s.shape[0]): 
      x[:,i]=u[:,i]/s[i]

   return x

def calcVnn(rn): 
   return 1./np.linalg.norm(rn[0]-rn[1])

def calcBareH(r,d,rb):
   t=calcT(d)
   v=calcVen(rb,r)

   return t+v

def loadNuCoord(): 
   r =setInitCoord()
   rb=setNucParameterForBasis()

   return r,rb

def calcX(d,n):
   x=np.zeros((n,n))

   s=calcS(d)
   e,u=np.linalg.eig(s)

   for i,v in enumerate(e): 
      x[:,i]=u[:,i]/ma.sqrt(v)

   return x

def calcX_THX(r,rb): 
   nd=rb.shape[0]

   d=calcDistance(rb)
   h=calcBareH(r,d,rb)
   x=calcX(d,nd)

   return x.T@h@x,x

def calcEnergy(h,x,r): 
   e,vec=np.linalg.eig(h)

   e +=calcVnn(r) 
   e  =np.sort(e)
   xm1=np.linalg.inv(x)
   vec=xm1@vec

   return e,vec

# <--- cython
def main(): 
   r,rb=loadNuCoord()
# test 
#   rb[0,2]=-1.25
#   rb[1,2]= 1.25
# test 
   x_thx,x=calcX_THX(r,rb)
   e_mo,vec_mo=calcEnergy(x_thx,x,r)

   print("=== Nuclear coordinate ====")
   print(r)
   print("==== Energy ====")
   print(e_mo)
   print("==== MO vector ====")
   print(vec_mo)

   return 

if __name__ == '__main__':
   main()
