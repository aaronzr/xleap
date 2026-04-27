import numpy as np
import copy

def integrate(data,coordinates):
    """
    Integrate takes an array of dimensions N1 x N2 x N3 x...NM and coordinate list of [[1xN1], [1xN2] ... [1xNm]]
    It integrates over all dimensions:
    """
    ndims=len(np.shape(data))
    for ii in range(ndims):
        data=np.trapz(data,coordinates[ii],axis=0)
    return data

def rmsdim_nonlinmesh(data,Coordinates=None):
    """
    rmsdim_nonlinmesh takes an array of dimensions N1 x N2 x N3 x...NM and calaculates the centroids and moment matrix.
    The coordinates may be nonlinear. If None, linear spacing is assumed.
    
    data: N1 x N2 x N3 x...NM
    Coordinates: M array of [[1xN1], [1xN2] ... [1xNm]] 
    *Note that I make a copy of Coordinates so that it doesn't change!
    
    returns: centroids,covariance
    Norm: scalar, integral over array (0th moment)
    Centroids: M x 1  array of centroids (1st moments)
    Covariance:  M x M array of moments (2nd moments)
    """
#   Make coordinates if not given
    ndims=len(np.shape(data))
    if Coordinates==None:
        coordinates=[];
        for ii in range(ndims):
            coordinates.append(np.array(range(np.shape(data)[ii])))
    else:
        coordinates=copy.copy(Coordinates)
        #coordinates=np.array(coordinates)
# init
    centroids=np.zeros([ndims])
    covariance=np.zeros([ndims,ndims])
#normalization
    norm=integrate(data,coordinates);
#centroids
    for ii in range(ndims):
        shape=np.ones(ndims,dtype=np.int8)
        shape[ii]=-1
        centroids[ii]=integrate(data*coordinates[ii].reshape(shape),coordinates)/norm
#Center
    for ii in range(ndims):
        coordinates[ii]=coordinates[ii]-centroids[ii]; #center everything
#Covariance
    for ii in range(ndims):
        ishape=np.ones(ndims,dtype=np.int8)
        ishape[ii]=-1
        x=coordinates[ii].reshape(ishape)
        for jj in range(ndims):
            jshape=np.ones(ndims,dtype=np.int8)
            jshape[jj]=-1
            y=coordinates[jj].reshape(jshape)
            covariance[ii,jj]=integrate((data*x)*y,coordinates)/norm
    return norm,centroids,covariance


def rmsmask(data,cent,cov,Coordinates=None,rmsfactor=1):
    """
    masks data based on rms ellipse (multiplied by rmsfactor)
    """
#   Make coordinates if not given
    ndims=len(np.shape(data))
    if Coordinates==None:
        coordinates=[];
        for ii in range(ndims):
            coordinates.append(np.array(range(np.shape(data)[ii])))
        coordinates=np.array(coordinates)
    else:
        coordinates=copy.copy(Coordinates)
#Center
    for ii in range(ndims):
        coordinates[ii]=coordinates[ii]-cent[ii]; #center everything
# make predicate
    icov=np.linalg.inv(cov*rmsfactor**2)
    def test(coords):
        return np.matmul(coords,np.matmul(icov,np.transpose(coords)))<1
# make mask
    mask=np.zeros(np.shape(data),dtype=np.int8)
    for index in np.ndindex(np.shape(mask)):
        coords=[coordinates[ii][index[ii]] for ii in range(len(index))]
        if test(coords):
            mask[index]=1
    return mask



def beam_mat(data,Xcoordinates=None,spec=[],Wcoordinates=[],xmask=[],wmask=[],autocenter=False,lam=2*np.pi):
    """
    Gets the beammat ("sigmas" of I) for a laser. 
    
    Calculates the "normal" terms (sigma_xx, sigma_xy, sigma_yy, sigma_wxwx, wigma_wxwy, sigma_wywy in 2D) from rmsdim_nonlinmesh
    
    Calculates the mixed terms for a laser (sigma_xx', sigma_xy', sigma_yx',sigma_yy' in 2D) directly in this routine
    The thing here is that we want [int(dw wF(xu(x))conjugate(F(u(x)))]), where the F(xu) is for the unprimed and wF(u) is for the primed coordinates.
    
    
    data: N1 x N2 x N3 x...NM .  . Normalized such that integrate(np.abs(data),xcoordinates)=1. (Code normalizes just in case)
    xcoordinates: M array of [[1xN1], [1xN2] ... [1xNm]]
    spec: fftn of data. Normalized such that integrate(np.abs(spec),xcoordinates)=1.(Code normalizes just in case)
    wcoordinates: fft coordiantes of xcoordinates (in angular frequnecy)
    xmask: mask used in the data domain (array of 1s an 0s with the same size as data)
    wmask: mask used inth fourier domian (array of 1s an 0s with the same size as spec, and thus data)
    lam: wavelength. Default of 2*pi sets k=1. With k=1 the "angular moments" will be in angle (instead of "momentum")
    autocenter: bool, if True the uses centered moments. Default is False.
    
    returns: beammat
    beammat: MxM matrix of effective beam moments (rms of intensity)
    """
    ndims=len(np.shape(data))
    beammat=np.zeros([ndims*2,ndims*2])
    k=2*np.pi/lam;
# make xcoordinates if neccesary
    if Xcoordinates==None:
        xcoordinates=[];
        for ii in range(ndims):
            x=np.array(range(np.shape(data)[ii]))
            xcoordinates.append(x-np.median(x))
    else:
        xcoordinates=copy.copy(Xcoordinates)

#normalize data, just in case
    data=data/np.sqrt(integrate(np.abs(data)**2,xcoordinates))
# make fft if it doesn't exist
    if (len(Wcoordinates)==0):
        wcoordinates=[];
        dV=1;
        for ii in range(ndims):
            dx=(np.median(np.diff(xcoordinates[ii])))
            dV=dV*dx
            wcoordinates.append(np.fft.fftshift(2*np.pi*np.fft.fftfreq(len(xcoordinates[ii]),dx)))
    else:
        dV=np.prod([np.median(np.diff(xcoordinates[ii])) for ii in range(ndims)])
        wcoordinates=copy.copy(Wcoordinates)
    if (len(spec)==0):
        spec=np.fft.fftshift(np.fft.fftn(np.fft.fftshift(data)))*dV/(np.sqrt(2*np.pi)**ndims)
    else:
        spec=spec/np.sqrt(integrate(np.abs(spec)**2,wcoordinates)) #normalize just in case
    #print(np.sqrt(integrate(np.abs(spec)**2,wcoordinates)))
# Copy coordiantes, since rms
#Intermediate terms
    I=np.abs(data)**2
    Iw=np.abs(spec)**2
#If no masks, then do something sensible:
    if len(xmask)==0:
        mask0=I>np.max(I)*0.01 #1.0% of peak value %
        __,cent,cov=rmsdim_nonlinmesh(I*mask0,xcoordinates)
        xmask=rmsmask(I,cent,cov,xcoordinates,rmsfactor=4)  #4 sigma cut
        xmask=xmask*mask0
    if len(wmask)==0:
        mask0=Iw>np.max(Iw)*0.01
        __,cent,cov=rmsdim_nonlinmesh(Iw*mask0,wcoordinates)
        wmask=rmsmask(Iw,cent,cov,wcoordinates,rmsfactor=4)
        wmask=wmask*mask0
#Autocenter  indicator
    center=1
    if autocenter:
        center=0;
#Get normal beammat terms
    __,xcent,xcov=rmsdim_nonlinmesh(I*xmask,xcoordinates)
    __,wcent,wcov=rmsdim_nonlinmesh(Iw*wmask,wcoordinates) 
    for ii in range(ndims):
        for jj in range(ndims):
            beammat[(ii*2,jj*2)]=xcov[(ii,jj)]+center*xcent[ii]*xcent[jj] # rmsdim_nonlinmesh centered the moments, we need to undo
            beammat[(jj*2,ii*2)]=xcov[(ii,jj)]+center*xcent[ii]*xcent[jj]
            beammat[(ii*2+1,jj*2+1)]=(wcov[(ii,jj)]+center*wcent[ii]*wcent[jj])/k**2
            beammat[(jj*2+1,ii*2+1)]=(wcov[(ii,jj)]+center*wcent[ii]*wcent[jj])/k**2
#Autocenter for angular terms
    if autocenter:
        for ii in range(ndims):
            xcoordinates[ii]=(xcoordinates[ii]-0*xcent[ii])*1; #center everything
            wcoordinates[ii]=wcoordinates[ii]-wcent[ii];
#Get the angular terms
    for ii in range(ndims):
        ishape=np.ones(ndims,dtype=np.int8)
        ishape[ii]=-1
        x=xcoordinates[ii].reshape(ishape)
        xF=np.fft.fftshift(np.fft.fftn(np.fft.fftshift(xmask*data*x)))*dV/(np.sqrt(2*np.pi)**ndims)
        for jj in range(ndims):
            jshape=np.ones(ndims,dtype=np.int8)
            jshape[jj]=-1
            w=wcoordinates[jj].reshape(jshape)
            if ii==jj:
                term=np.real(1j+2*integrate((xF)*(np.conj(wmask*spec)*w),wcoordinates))
            else:
                term=np.real(2*integrate((xF)*(np.conj(wmask*spec)*w),wcoordinates))
            beammat[(ii)*2,(jj)*2+1]=term/k/2
            beammat[(jj)*2+1,(ii)*2]=term/k/2
    return beammat


def peak_fwhm(pk_idx,lis,thresh=None,outfrompeak=True,interp=False):
    """
    pk_idx: integer index of the peak in list
    lis: the list of values
    thresh: the value of the threshold. An absolute number. Default is max(list)/2 which would give FWM
    outfrompeak: bool choice whether to search starting from peak, or from edges of list
    interp: bool choice whether to interpolate value between last two points.
    
    Note search is exclusice: so values found will be at or lower than Thresh (rather than last "Good" point)
    returns: pt1,pt2
    """
    m=lis[pk_idx]
    if thresh==None:
        thresh=m/2
    pt1=pk_idx;pt2=pk_idx;
    if outfrompeak:
        try:
            for idx,l in enumerate(np.flip(lis[0:pk_idx])):
                if l<=thresh:
                    pt1=pk_idx-idx-1;
                    break
        except:
            pt1=0
        try:
            for idx,l in enumerate(lis[pk_idx:]):
                if l<=thresh:
                    pt2=pk_idx+idx;
                    break
        except:
            pt2=len(lis)
    else:
        try:
            for idx,l in enumerate(lis[0:pk_idx]):
                if l>=thresh:
                    pt1=idx-1;
                    break
        except:
            pt1=0
        try:
            for idx,l in enumerate(np.flip(lis[pk_idx:])):
                if l>=thresh:
                    pt2=len(lis)-idx;
                    break
        except:
            pt2=len(lis)
        pt1=np.clip(pt1,1,len(lis)-2)
        pt2=np.clip(pt2,1,len(lis)-2)
    #print(lis[pt1],lis[pt2])
    if interp:
        try:
            invweights=np.abs(np.array(lis[pt1:pt1+2])-thresh)
            invweights[invweights==0]=1e5;
            weights=1/invweights
            pt1=np.sum(np.array([pt1,pt1+1]*weights)/np.sum(weights))
        except:
            pt1=pt1
        try:
            invweights=np.abs(np.array(lis[pt2-1:pt2+1])-thresh)
            invweights[invweights==0]=1e-5;
            weights=1/invweights
            pt2=np.sum(np.array([pt2-1,pt2]*weights)/np.sum(weights))
        except:
            pt2=pt2
        pt1=np.clip(pt1,1,len(lis)-2)
        pt2=np.clip(pt2,1,len(lis)-2)
    return pt1,pt2