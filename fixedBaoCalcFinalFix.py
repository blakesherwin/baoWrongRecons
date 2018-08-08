import nbodykit
import pylab
import numpy
import pickle
import matplotlib
from nbodykit.lab import cosmology
from scipy.interpolate import CubicSpline

### two possible problems: smoothing scale and other.### check damping is ok.

### basic tools
def kxkzToKmu(kx,kz):
    k = numpy.sqrt(kx**2. + kz**2.) 
    mu = kz/k
    return k, mu

def kmuToKxkz(k,mu):
    kz = k*mu
    kx = numpy.sqrt(1.-mu**2.)*k
    return kx, kz

def S(kx,kz=None,ksmooth=0.1):
    if type(kz) is numpy.ndarray:
        kz = kz
    else:
        kz = 0.#kx
    Ss = numpy.arange(len(kx))
    Ss = Ss*0.00
    Ss = Ss+1.00    
    return Ss*numpy.exp(-(kx**2.+kz**2.)/ksmooth**2./2.)### for 0.1, this is equivalent to smoothing scale of 14.1/h mpc as defined in seo et al 2015.

### damping coefficient calculation
def cSigmaSWrongPerp(k,kdiff,p,b,f,btilde,ftilde,alphaPar,alphaPerp):
    length = len(k)
    SigmaSMu = k
    SigmaSMu = SigmaSMu *0.
    muv = (numpy.arange(200.)-100.)/100.
    deltaMu = 0.01
    
    for i in xrange(length):
        kv = k[i]
        kxAinv = kv*numpy.sqrt(1.-muv**2.)/alphaPerp
        kzAinv = kv*muv/alphaPar
        kAinv = numpy.sqrt(kxAinv**2.+kzAinv**2.)
        muAinv = numpy.nan_to_num(kzAinv/kAinv)
        g = (b+f*muv**2.)/(btilde+ftilde*muAinv**2.)
        SigmaSMu[i] = numpy.sum(1./(2.*numpy.pi)**3.*deltaMu*2.*numpy.pi*kv**2.*(1-muv**2.)*kv**2.*g**2.*S(kxAinv,kzAinv)**2.*p[i]/kv**4./2./alphaPerp**4.)
    return numpy.sum(SigmaSMu*kdiff)

def cSigmaSWrongPar(k,kdiff,p,b,f,btilde,ftilde,alphaPar,alphaPerp):
    length = len(k)
    SigmaSMu = k
    SigmaSMu = SigmaSMu *0.
    muv = (numpy.arange(200.)-100.)/100.
    deltaMu = 0.01
    
    for i in xrange(length):
        kv = k[i]
        kxAinv = kv*numpy.sqrt(1.-muv**2.)/alphaPerp
        kzAinv = kv*muv/alphaPar
        kAinv = numpy.sqrt(kxAinv**2.+kzAinv**2.)
        muAinv = numpy.nan_to_num(kzAinv/kAinv)
        g = (b+f*muv**2.)/(btilde+ftilde*muAinv**2.)
        SigmaSMu[i] = numpy.sum(1./(2.*numpy.pi)**3.*deltaMu*2.*numpy.pi*kv**2.*(muv**2.)*kv**2.*g**2.*S(kxAinv,kzAinv)**2.*p[i]/kv**4./alphaPar**4.)
    return numpy.sum(SigmaSMu*kdiff)

def cSigmaDWrongPerp(k,kdiff,p,b,f,btilde,ftilde,alphaPar,alphaPerp):
    length = len(k)
    SigmaSMu = k
    SigmaSMu = SigmaSMu *0.
    muv = (numpy.arange(200.)-100.)/100.
    deltaMu = 0.01
    
    for i in xrange(length):
        kv = k[i]
        kxAinv = kv*numpy.sqrt(1.-muv**2.)/alphaPerp
        kzAinv = kv*muv/alphaPar
        kAinv = numpy.sqrt(kxAinv**2.+kzAinv**2.)
        muAinv = numpy.nan_to_num(kzAinv/kAinv)
        g = (b+f*muv**2.)/(btilde+ftilde*muAinv**2.)
        SigmaSMu[i] = numpy.sum(1./(2.*numpy.pi)**3.*deltaMu*2.*numpy.pi*kv**2.*(1-muv**2.)*kv**2.*(1.-g*S(kxAinv,kzAinv)/alphaPerp**2.)**2.*p[i]/kv**4./2.)
    return numpy.sum(SigmaSMu*kdiff)

def cSigmaDWrongPar(k,kdiff,p,b,f,btilde,ftilde,alphaPar,alphaPerp):
    length = len(k)
    SigmaSMu = k
    SigmaSMu = SigmaSMu *0.
    muv = (numpy.arange(200.)-100.)/100.
    deltaMu = 0.01
    
    for i in xrange(length):
        kv = k[i]
        kxAinv = kv*numpy.sqrt(1.-muv**2.)/alphaPerp
        kzAinv = kv*muv/alphaPar
        kAinv = numpy.sqrt(kxAinv**2.+kzAinv**2.)
        muAinv = numpy.nan_to_num(kzAinv/kAinv)
        g = (b+f*muv**2.)/(btilde+ftilde*muAinv**2.)
        SigmaSMu[i] = numpy.sum(1./(2.*numpy.pi)**3.*deltaMu*2.*numpy.pi*kv**2.*(muv**2.)*kv**2.*(1.-(1.+ftilde)/(1.+f)*g*S(kxAinv,kzAinv)/alphaPar**2.)**2.*p[i]/kv**4./2.)
    return numpy.sum(SigmaSMu*kdiff)

def cSigmaSDWrongPerp(k,kdiff,p,b,f,btilde,ftilde,alphaPar,alphaPerp):
    SigmaSD = cSigmaSWrongPerp(k,kdiff,p,b,f,btilde,ftilde,alphaPar,alphaPerp)+cSigmaDWrongPerp(k,kdiff,p,b,f,btilde,ftilde,alphaPar,alphaPerp)
    SigmaSD =  SigmaSD/2.
    return SigmaSD

def cSigmaSDWrongPar(k,kdiff,p,b,f,btilde,ftilde,alphaPar,alphaPerp):
    SigmaSD = cSigmaSWrongPar(k,kdiff,p,b,f,btilde,ftilde,alphaPar,alphaPerp)+cSigmaDWrongPar(k,kdiff,p,b,f,btilde,ftilde,alphaPar,alphaPerp)
    SigmaSD =  SigmaSD/2.
    return SigmaSD

### reconstructed power spectrum (general) + squashed power spectrum (fixed alpha)
def totalPowerWrongRecons(kk,mu,b,f,btilde,ftilde,alphaPar,alphaPerp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p):
    kx, kz = kmuToKxkz(kk,mu)
    kxPrimed = kx * alphaPerp
    kzPrimed = kz * alphaPar
    kPrimed = numpy.sqrt(kxPrimed**2.+kzPrimed**2.)
    muPrimed = kzPrimed/kPrimed
    muPrimed = numpy.nan_to_num(muPrimed)

    ModS = S(kx,kz)*(1.+ftilde*mu**2.)/(btilde+ftilde*mu**2.)
    ssExp = -kPrimed**2.*((1.-mu**2.)*SigmaSPerp+(1.+ftilde)**2.*mu**2.*SigmaSPar)
    ddExp = -kPrimed**2.*((1.-mu**2.)*SigmaDPerp+(1.+f)**2.*mu**2.*SigmaDPar)
    sdExp = ssExp*0.5+ddExp*0.5
    totalPower = alphaPar*alphaPerp**2.*(b+f*muPrimed**2.)**2.*numpy.interp(kPrimed,k,p)*(ModS**2.*numpy.exp(ssExp)+(1.-ModS)**2.*numpy.exp(ddExp)+2.*ModS*(1-ModS)*numpy.exp(sdExp))
    return totalPower

def totalPowerWrongReconsFixedAlpha(kk,mu,b,f,btilde,ftilde,alphaPar,alphaPerp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p):
    kx, kz = kmuToKxkz(kk,mu)
    kxPrimed = kx * alphaPerp
    kzPrimed = kz * alphaPar
    kx = kxPrimed.copy()
    kz = kzPrimed.copy()

    kPrimed = numpy.sqrt(kxPrimed**2.+kzPrimed**2.)
    muPrimed = kzPrimed/kPrimed
    kk = kPrimed.copy()
    mu = muPrimed.copy()#### is this wrong? no it is ok.
    muPrimed = numpy.nan_to_num(muPrimed)

    ModS = S(kx,kz)*(1.+ftilde*mu**2.)/(btilde+ftilde*mu**2.)
    ssExp = -kPrimed**2.*((1.-mu**2.)*SigmaSPerp+(1.+ftilde)**2.*mu**2.*SigmaSPar)
    ddExp = -kPrimed**2.*((1.-mu**2.)*SigmaDPerp+(1.+f)**2.*mu**2.*SigmaDPar)
    sdExp = ssExp*0.5+ddExp*0.5
    totalPower = alphaPar*alphaPerp**2.*(b+f*muPrimed**2.)**2.*numpy.interp(kPrimed,k,p)*(ModS**2.*numpy.exp(ssExp)+(1.-ModS)**2.*numpy.exp(ddExp)+2.*ModS*(1-ModS)*numpy.exp(sdExp))
    return totalPower

### monopole, quadrupole etc
def totalPowerMonopoleWrong(kk,b,f,btilde,ftilde,alphaPar,alphaPerp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p):
    try:
        length = len(kk)
        monopole = kk
    except:
        monopole = numpy.array([kk])
        print 'exception'
        length = 0
    monopole = monopole *0.

    muv = (numpy.arange(200.)-100.)/100.
    deltaMu = 0.01
    for i in xrange(length):
        kv = kk[i]
        monopole[i] = 0.5* numpy.sum(totalPowerWrongRecons(kv,muv,b,f,btilde,ftilde,alphaPar,alphaPerp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p)*deltaMu)
    return monopole

def totalPowerMonopoleWrongFixedAlpha(kk,b,f,btilde,ftilde,alphaPar,alphaPerp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p):
    try:
        length = len(kk)
        monopole = kk
    except:
        monopole = numpy.array([kk])
        print 'exception'
        length = 0
    monopole = monopole *0.

    muv = (numpy.arange(200.)-100.)/100.
    deltaMu = 0.01
    for i in xrange(length):
        kv = kk[i]
        monopole[i] = 0.5* numpy.sum(totalPowerWrongReconsFixedAlpha(kv,muv,b,f,btilde,ftilde,alphaPar,alphaPerp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p)*deltaMu)
    return monopole

def totalPowerQuadrupoleWrong(kk,b,f,btilde,ftilde,alphaPar,alphaPerp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p):
    try:
        length = len(kk)
        quadrupole = kk
    except:
        quadrupole = numpy.array([kk])
        print 'exception'
        length = 0
    quadrupole = quadrupole *0.

    muv = (numpy.arange(200.)-100.)/100.
    deltaMu = 0.01
    for i in xrange(length):
        kv = kk[i]
        quadrupole[i] = 5./2.*numpy.sum((3./2.*muv**2.-1./2.)*totalPowerWrongRecons(kv,muv,b,f,btilde,ftilde,alphaPar,alphaPerp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p)*deltaMu)
    return quadrupole

def totalPowerQuadrupoleWrongFixedAlpha(kk,b,f,btilde,ftilde,alphaPar,alphaPerp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p):
    try:
        length = len(kk)
        quadrupole = kk
    except:
        quadrupole = numpy.array([kk])
        print 'exception'
        length = 0
    quadrupole = quadrupole *0.

    muv = (numpy.arange(200.)-100.)/100.
    deltaMu = 0.01
    for i in xrange(length):
        kv = kk[i]
        quadrupole[i] = 5./2.*numpy.sum((3./2.*muv**2.-1./2.)*totalPowerWrongReconsFixedAlpha(kv,muv,b,f,btilde,ftilde,alphaPar,alphaPerp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p)*deltaMu)
    return quadrupole








####### main code ##

### set up cosmology and power spectra
cosmo = cosmology.Planck15
P = cosmology.power.linear.LinearPower(cosmo, 0., transfer='EisensteinHu')
PNW = cosmology.power.linear.LinearPower(cosmo, 0., transfer='NoWiggleEisensteinHu')
m = numpy.loadtxt('matterpower.dat')
k = m[:,0]
pickle.dump(k,open('kvec.pkl','w'))
k1 = numpy.roll(k,1)
kdiff = k-k1
kdiff[0] = 0.
p = numpy.array(P(k))#m[:,1]
pn = numpy.array(PNW(k))
corr = nbodykit.cosmology.correlation.pk_to_xi(k,p)
rrr = numpy.arange(1000.)/1000.*200.
pylab.plot(rrr,corr(rrr)*rrr**2.)
print 'peak input 0 ', rrr[rrr**2.*corr(rrr)==numpy.max((rrr**2.*corr(rrr))[rrr>80.])]
pylab.axvline(x=98.6,color='k',linestyle=':')
pylab.savefig('correlationPlot.png')
pylab.clf()


### do not modify!!! default values of parameters
awrongfactor = 1.03
bf = 2.
ff = 0.55
afperp = 1.
afpar = 1.

#### calculate damping exponents
SigmaSPerp = cSigmaSWrongPerp(k,kdiff,p,2.,0.55,bf,ff,afpar,afperp)
SigmaSPar = cSigmaSWrongPar(k,kdiff,p,2.,0.55,bf,ff,afpar,afperp)
SigmaDPerp = cSigmaDWrongPerp(k,kdiff,p,2.,0.55,bf,ff,afpar,afperp)
SigmaDPar = cSigmaDWrongPar(k,kdiff,p,2.,0.55,bf,ff,afpar,afperp)

#### evaluate default and 'squashed' spectra
defaultMonopole = (totalPowerMonopoleWrong(k,2.,0.55,bf,ff,afpar,afperp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p)-totalPowerMonopoleWrong(k,2.,0.55,bf,ff,afpar,afperp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,pn))/PNW(k)
defaultQuadrupole = (totalPowerQuadrupoleWrong(k,2.,0.55,bf,ff,afpar,afperp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p)-totalPowerQuadrupoleWrong(k,2.,0.55,bf,ff,afpar,afperp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,pn))/PNW(k)
defaultMonopoleFixedAlphaPerp = (totalPowerMonopoleWrongFixedAlpha(k,2.,0.55,bf,ff,awrongfactor,afpar,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p)-totalPowerMonopoleWrongFixedAlpha(k,2.,0.55,bf,ff,awrongfactor,afpar,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,pn))/PNW(k)
defaultMonopoleFixedAlphaPar = (totalPowerMonopoleWrongFixedAlpha(k,2.,0.55,bf,ff,afperp,awrongfactor,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p)-totalPowerMonopoleWrongFixedAlpha(k,2.,0.55,bf,ff,afperp,awrongfactor,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,pn))/PNW(k)

correlationMonopoleDefault = nbodykit.cosmology.correlation.pk_to_xi(k,totalPowerMonopoleWrong(k,2.,0.55,bf,ff,afpar,afperp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p)\
)
correlationQuadrupoleDefault = nbodykit.cosmology.correlation.pk_to_xi_quad(k,totalPowerQuadrupoleWrong(k,2.,0.55,bf,ff,afpar,afperp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p)\
)
correlationMonopoleWrongPar = nbodykit.cosmology.correlation.pk_to_xi(k,totalPowerMonopoleWrongFixedAlpha(k,2.,0.55,bf,ff,awrongfactor,afpar,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p))
correlationMonopoleWrongPerp = nbodykit.cosmology.correlation.pk_to_xi(k,totalPowerMonopoleWrongFixedAlpha(k,2.,0.55,bf,ff,afperp,awrongfactor,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p))
correlationQuadrupoleWrongPar = nbodykit.cosmology.correlation.pk_to_xi_quad(k,totalPowerQuadrupoleWrongFixedAlpha(k,2.,0.55,bf,ff,awrongfactor,afperp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p))
correlationQuadrupoleWrongPerp = nbodykit.cosmology.correlation.pk_to_xi_quad(k,totalPowerQuadrupoleWrongFixedAlpha(k,2.,0.55,bf,ff,afpar,awrongfactor,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p))


factorList = [1.1,1.3,awrongfactor,awrongfactor]
for i in xrange(4):
    factorListDefault = [1.,1.,1.,1.]
    factorListDefault[i] =  factorList[i]
    bf = 1.*2.*factorListDefault[0]
    ff = 1.*0.55*factorListDefault[1]
    afperp = 1.*factorListDefault[2]
    afpar = 1.*factorListDefault[3]
    print 'running calculation with: b=', bf, ', f=', ff, ', aperp=', afperp, ', apar=', afpar
     
    ### calculate damping exponents. note the arguments are alpha par alpha perp!
    SigmaSPerp = cSigmaSWrongPerp(k,kdiff,p,2.,0.55,bf,ff,afpar,afperp)
    SigmaSPar = cSigmaSWrongPar(k,kdiff,p,2.,0.55,bf,ff,afpar,afperp)
    SigmaDPerp = cSigmaDWrongPerp(k,kdiff,p,2.,0.55,bf,ff,afpar,afperp)
    SigmaDPar = cSigmaDWrongPar(k,kdiff,p,2.,0.55,bf,ff,afpar,afperp)

    ### calculate monopoles and quadrupoles, first in fourier then real space
    newMonopole = (totalPowerMonopoleWrong(k,2.,0.55,bf,ff,afpar,afperp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p)-totalPowerMonopoleWrong(k,2.,0.55,bf,ff,afpar,afperp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,pn))/PNW(k)
    newQuadrupole = (totalPowerQuadrupoleWrong(k,2.,0.55,bf,ff,afpar,afperp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p)-totalPowerQuadrupoleWrong(k,2.,0.55,bf,ff,afpar,afperp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,pn))/PNW(k)

    correlationMonopoleWrong = nbodykit.cosmology.correlation.pk_to_xi(k,totalPowerMonopoleWrong(k,2.,0.55,bf,ff,afpar,afperp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p))
    correlationQuadrupoleWrong = nbodykit.cosmology.correlation.pk_to_xi_quad(k,totalPowerQuadrupoleWrong(k,2.,0.55,bf,ff,afpar,afperp,SigmaSPar,SigmaSPerp,SigmaDPar,SigmaDPerp,p))

    #### derive shifts in a few different (approximate) ways.
    khi = numpy.arange(100000.)/100000.
    newMonopole2 = CubicSpline(k,newMonopole)
    defaultMonopole2 = CubicSpline(k,defaultMonopole)
    if afperp >1:
        defaultMonopoleFix2 = CubicSpline(k,defaultMonopoleFixedAlphaPerp)
        correlationMonopoleSquash = correlationMonopoleWrongPerp
        correlationQuadrupoleSquash = correlationQuadrupoleWrongPerp
    elif afpar >1: 
        defaultMonopoleFix2 = CubicSpline(k,defaultMonopoleFixedAlphaPar)
        correlationMonopoleSquash = correlationMonopoleWrongPar
        correlationQuadrupoleSquash = correlationQuadrupoleWrongPar
    else:
        defaultMonopoleFix2 = defaultMonopole2
        correlationMonopoleSquash = correlationMonopoleDefault
        correlationQuadrupoleSquash = correlationQuadrupoleDefault

    newMonopolehi = newMonopole2(khi)
    defaultMonopolehi = defaultMonopole2(khi)
    defaultMonopoleFixhi = defaultMonopoleFix2(khi)

    kcrit = 0.25
    maxKOfWrong =  khi[newMonopolehi==numpy.max(newMonopolehi[khi>kcrit])]
    maxKOfDefault =  khi[defaultMonopolehi==numpy.max(defaultMonopolehi[khi>kcrit])]
    maxKOfRight = khi[defaultMonopoleFixhi==numpy.max(defaultMonopoleFixhi[khi>kcrit])]
    print 'high res', khi[newMonopolehi==numpy.max(newMonopolehi)]/ khi[defaultMonopolehi==numpy.max(defaultMonopolehi)]
    print 'high res relative to squashed one',  khi[newMonopolehi==numpy.max(newMonopolehi)]/ khi[defaultMonopoleFixhi==numpy.max(defaultMonopoleFixhi)]*0.5+khi[newMonopolehi==numpy.min(newMonopolehi)]/ khi[defaultMonopoleFixhi==numpy.min(defaultMonopoleFixhi)]*0.5
    print 'redoing with higher peaks above 0.15, first highres then squash', maxKOfWrong/maxKOfDefault, maxKOfWrong/maxKOfRight

    rArray = numpy.arange(1000000.)/1000000.*200.
    wrongCorr = correlationMonopoleWrong(rArray)
    squashCorr = correlationMonopoleSquash(rArray)
    defaultCorr = correlationMonopoleDefault(rArray)
    wrongQuad = correlationQuadrupoleWrong(rArray)
    squashQuad = correlationQuadrupoleSquash(rArray)
    defaultQuad = correlationQuadrupoleDefault(rArray)

    maximumOfWrong = rArray[numpy.max(wrongCorr[rArray>80.]*rArray[rArray>80.]**2.)==wrongCorr*rArray**2.]
    maximumOfSquash = rArray[numpy.max(squashCorr[rArray>80.]*rArray[rArray>80.]**2.)==squashCorr*rArray**2.]
    maximumOfDefault = rArray[numpy.max(defaultCorr[rArray>80.]*rArray[rArray>80.]**2.)==defaultCorr*rArray**2.]

    print 'real space location', maximumOfDefault, 'ratio of wrong to squashed',maximumOfWrong/maximumOfSquash, '(+ ratio to default', maximumOfWrong/maximumOfDefault, ')'


    ########## begin plotting #################
    ### fourier space ####
    # ratio
    pylab.clf()
    pylab.plot(k[k<0.4],(newMonopole/defaultMonopole)[k<0.4],label='mono. ratio',color='b')
    pylab.plot(k[k<0.4],(newQuadrupole/defaultQuadrupole)[k<0.4],label='quad. ratio',color='r')
    pylab.plot(khi[khi<0.4],(newMonopolehi/defaultMonopolehi)[khi<0.4],label='mono. ratio hi')
    pylab.xlabel('k[h/Mpc]')
    pylab.ylabel('wrong/correct recons P(k)[(Mpc/h)^3]')
    pylab.legend(loc='best')
    pylab.ylim(0.9,1.2)
    pylab.savefig('finalPowerSpecRatioApp'+str(int(afperp*100))+str(int(afpar*100))+'BiasRSD'+str(int(bf*100))+str(int(ff*100))+'.eps',bbox_inches='tight')
    pylab.savefig('finalPowerSpecRatioApp'+str(int(afperp*100))+str(int(afpar*100))+'BiasRSD'+str(int(bf*100))+str(int(ff*100))+'.png',bbox_inches='tight')
    # difference
    pylab.clf()
    pylab.plot(k[k<0.4],(newMonopole-defaultMonopole)[k<0.4],label='monopole diff.',color='b')
    pylab.plot(k[k<0.4],(newQuadrupole-defaultQuadrupole)[k<0.4],label='quadrupole diff.',color='r')
    pylab.plot(k[k<0.4],(0.03*defaultMonopole)[k<0.4],label='0.03 '+r'$\times$'+' default monopole ',linestyle=':',color='k')
    pylab.xlabel(r'$k \ [\mathrm{h/Mpc}]$',fontsize=14)
    pylab.ylabel(r'$\Delta P(k)[(\mathrm{Mpc/h})^3]$',fontsize=14)
    pylab.legend(loc='best')
    pylab.savefig('finalPowerSpecDiffApp'+str(int(afperp*100))+str(int(afpar*100))+'BiasRSD'+str(int(bf*100))+str(int(ff*100))+'.eps',bbox_inches='tight')    
    pylab.savefig('finalPowerSpecDiffApp'+str(int(afperp*100))+str(int(afpar*100))+'BiasRSD'+str(int(bf*100))+str(int(ff*100))+'.png',bbox_inches='tight')    

    #### configuration space ####
    #difference
    pylab.clf()
    pylab.plot(rArray,rArray**2.*((wrongCorr-defaultCorr)*3.+defaultCorr),label='wrong corr. (diff. x 3)',color='b')
    pylab.plot(rArray,rArray**2.*((squashCorr-defaultCorr)*3.+defaultCorr),label='rescaled corr. (diff. x 3)',linestyle='--',color='b')
    pylab.plot(rArray,rArray**2.*defaultCorr,label='default corr.',linestyle=':',color='b')
    pylab.plot(rArray,-rArray**2.*(wrongQuad),label='wrong quad.',color='r',alpha=1.)
    pylab.plot(rArray,-rArray**2.*(squashQuad),label='rescaled quad.',color='r',alpha=1.,linestyle='--')
    pylab.plot(rArray,-rArray**2.*(defaultQuad),label='default quad.',linestyle=':',color='r',alpha=1.)

    pylab.xlabel('separation '+r'$r  \ \mathrm{[Mpc / h]}$',fontsize=14)
    pylab.ylabel('correlation function '+r'$\xi(r)$',fontsize=14)
    pylab.legend(loc='best')
    pylab.savefig('finalCorrelationDiffApp'+str(int(afperp*100))+str(int(afpar*100))+'BiasRSD'+str(int(bf*100))+str(int(ff*100))+'.png',bbox_inches='tight')
    pylab.savefig('finalCorrelationDiffApp'+str(int(afperp*100))+str(int(afpar*100))+'BiasRSD'+str(int(bf*100))+str(int(ff*100))+'.eps',bbox_inches='tight')
    # ratio
    pylab.clf()
    pylab.plot(rArray,wrongCorr/squashCorr,label='wrong / rescaled corr.',color='b')
    pylab.plot(rArray[rArray>10.],((wrongQuad/squashQuad)[rArray>10.]-1.)/5.+1.,label='wrong / rescaled quad. '+r'$\times 0.2$',color='r')
    pylab.axvline(x=maximumOfDefault,color='k',linestyle=':')
    pylab.xlabel('separation '+r'$r  \ \mathrm{[Mpc / h]}$',fontsize=14)
    pylab.ylabel('ratio of correlation functions '+r'$\xi(r)$',fontsize=14 )
    pylab.xlim(0,115.)
    pylab.ylim(0.97,1.03)
    if (afperp!=afpar):
        pylab.ylim(0.985,1.015)
    pylab.legend(loc='best')
    pylab.savefig('finalCorrelationRatioApp'+str(int(afperp*100))+str(int(afpar*100))+'BiasRSD'+str(int(bf*100))+str(int(ff*100))+'.png',bbox_inches='tight')
    pylab.savefig('finalCorrelationRatioApp'+str(int(afperp*100))+str(int(afpar*100))+'BiasRSD'+str(int(bf*100))+str(int(ff*100))+'.eps',bbox_inches='tight')

