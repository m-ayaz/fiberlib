# -*- coding: utf-8 -*-
"""
Created on Thu May  7 16:40:43 2020

@author: Mostafa
"""

#import time

import matplotlib.pyplot as plt

from pickle import load as pickleload

from tqdm import tqdm

from numpy import cos,pi,exp,conj,sqrt,array,real,var,log10,mean,floor,ones,\
kron,arcsinh,arange,roll,reshape,ceil,sum,abs,cumsum,cumprod,zeros,prod,angle,imag,average
#from numpy import sin,cos,pi,log,exp,conj,tan,sqrt,array,real,imag,var,log10\
#,mean,linspace,floor,ones,sinc,kron,matmul,arcsinh,arange,roll,concatenate\
#,reshape,ceil,sum,abs,cumsum,cumprod,zeros,prod,angle

from numpy.random import uniform,randint,randn,choice
#from numpy.random import rand,uniform,randint,randn

from numpy.fft import fft,ifft

from commpy.filters import rrcosfilter
#from commpy.filters import rcosfilter,rrcosfilter

from scipy.signal import resample_poly,upfirdn
#from scipy.signal import resample_poly,upfirdn,resample,remez

from scipy.special import sici
#%%
__ISREMOVEDTRANSIENT__=True
#%% Test Functions Area
#def potsig(inputSignal):
#    return mean(sum(abs(inputSignal)**2,0))
#%%
class D4:
    
    def __init__(self):
        mod_file=open('Modulations_Alphabets.pickle','rb')
        ModulationDict_temp=pickleload(mod_file)
        mod_file.close()
        self.ModPowerDict={modulation:mean(sum(abs(ModulationDict_temp[modulation])**2,1)) for modulation in ModulationDict_temp}
        self.ModulationDict=ModulationDict_temp
        
    def getModPhiPsi(self,modname):
        ''' To be fixed! '''
        ModAlphabet=self.ModulationDict[modname]
        Phi_value=mean(sum(abs(ModAlphabet)**4,1))/mean(sum(abs(ModAlphabet)**2,1))**2-2
        Psi_value=mean(sum(abs(ModAlphabet)**4,1))/mean(sum(abs(ModAlphabet)**2,1))**2-2
        return Phi_value,Psi_value
        
    def __str__(self):
        return 'Available modulation formats:\n\n'+'\n'.join(sorted(self.ModulationDict))
    
#    def scatplot(self,mod_name):
#        try:
#            mod_alphabet=self.ModulationDict[mod_name]
#            plt.figure()
##            plt.plot(real(mod_alphabet[0]),imag(mod_alphabet[0]))
#            plt.figure()
##            plt.plot(real(mod_alphabet[1]),imag(mod_alphabet[1]))
#            plt.figure()
#        except:
#            print('No such modulation!')
#        return 'Available modulation formats:\n\n'+'\n-----------------------\n'.join(self.ModulationDict)
#%%
def RRC_f(f,SymbolRate,roll_off_factor):
    '''
    Pulse Shape; for now, it is assumed to be
    rectangular in frequency
    '''
    pass_band=abs(f)<SymbolRate*(1-roll_off_factor)/2
    
    if roll_off_factor==0:
        transient_band=0
    else:
        transient_band=cos(pi/2/roll_off_factor/SymbolRate*(abs(f)-SymbolRate*(1-roll_off_factor)/2))*\
        (abs(f)>=SymbolRate*(1-roll_off_factor)/2)*(abs(f)<SymbolRate*(1+roll_off_factor)/2)
    
    return (pass_band+transient_band)/SymbolRate
    
#    if ps_type=='rect':
#        return (f<SymbolRate/2)*(f>-SymbolRate/2)/SymbolRate
#    else:
#        return 0
#%%
''''''''''''''''''
''''''''''''''''''
''''''''''''''''''
'''''''''''''''
Complete MonteCarlo integration functions
may be found in fiberlib backups!
'''''''''''''''
''''''''''''''''''
''''''''''''''''''
''''''''''''''''''
def _2DMC(func,x0,x1,y0,y1,_N=10000):
    '''2D Monte Carlo Integration'''
    _N=int(_N)
    x=uniform(x0,x1,_N)
    y=uniform(y0,y1,_N)
    return sum(func(x,y))*(x1-x0)*(y1-y0)/_N
#%%
def _3DMC(func,x0,x1,y0,y1,z0,z1,_N=10000):
    '''3D Monte Carlo Integration'''
    _N=int(_N)
    x=uniform(x0,x1,_N)
    y=uniform(y0,y1,_N)
    z=uniform(z0,z1,_N)
    return sum(func(x,y,z))*(x1-x0)*(y1-y0)*(z1-z0)/_N
#%%
def _4DMC(func,w0,w1,x0,x1,y0,y1,z0,z1,_N=10000):
    '''4D Monte Carlo Integration'''
    _N=int(_N)
    w=uniform(w0,w1,_N)
    x=uniform(x0,x1,_N)
    y=uniform(y0,y1,_N)
    z=uniform(z0,z1,_N)
    return sum(func(w,x,y,z))*(w1-w0)*(x1-x0)*(y1-y0)*(z1-z0)/_N
#%%
def _5DMC(func,v0,v1,w0,w1,x0,x1,y0,y1,z0,z1,_N=10000):
    '''5D Monte Carlo Integration'''
    _N=int(_N)
    v=uniform(v0,v1,_N)
    w=uniform(w0,w1,_N)
    x=uniform(x0,x1,_N)
    y=uniform(y0,y1,_N)
    z=uniform(z0,z1,_N)
    return sum(func(v,w,x,y,z))*(v1-v0)*(w1-w0)*(x1-x0)*(y1-y0)*(z1-z0)/_N
#%%
def Upsilon(link_index,f1,f2,f,alpha,beta2,gamma,Lspan,Nspan,TotalAccDispersionOverLinks,i1,it,i2,NumLinks):
    '''
    "f1" and "f2" must have the same size i.e.
    they must be both vectors or matrices.
    
    "link_index" is the same "k" in the main model equations.
    '''
    if gamma==0:
        return 0
    
    theta_beta=4*pi**2*beta2*(f1-f)*(f2-f)
    
    if alpha<1e-6:
        raise Exception('Too small fiber attenuation')
    
    temp1=(1-exp(-alpha*Lspan+1j*Lspan*theta_beta))/(alpha-1j*theta_beta)
    
    if beta2==0:
        temp2=Nspan
    else:
        temp2=(exp(1j*Nspan*Lspan*theta_beta)-1)/(exp(1j*Lspan*theta_beta)-1)
        
    temp3=exp(-2j*pi**2*Nspan*beta2*Lspan*f**2)
    
#    print('\n\n',temp3,'\n\n')
    
    Psi_term=gamma*temp1*temp2*temp3
    '''==============================================='''
    tempUpsilon=exp(
            1j*2*pi**2*f1**2*TotalAccDispersionOverLinks[i1,link_index]-
            1j*2*pi**2*(f1+f2-f)**2*TotalAccDispersionOverLinks[it,link_index]+
            1j*2*pi**2*f2**2*TotalAccDispersionOverLinks[i2,link_index]+
            1j*2*pi**2*f**2*TotalAccDispersionOverLinks[link_index+1,NumLinks]
            )
    
#    print(tempUpsilon)
    
#    tempUpsilon=1
    
#    print(tempUpsilon)
    
#    tempUpsilon=temp3=1
    
#    print(tempUpsilon)
    
    return tempUpsilon*Psi_term
#%%
def SNR_EGN_noncoh(EGN_noncoh_Accessories,NLI_terms=None,_NMC=10e6,printlog=False):
    '''This function works only when all the fiber losses are totally
    compensated by amplifiers.'''
    
    one_to_SNR_Linear = 0 # 1/SNR
    
    for iLink_EGN_Accessories in EGN_noncoh_Accessories:
#        SNR_SingleLink_dB,_,__,___=SNR_EGN(iLink_EGN_Accessories,NLI_terms,_NMC,printlog)
        SNR_SingleLink_dB=SNR_EGN(iLink_EGN_Accessories,NLI_terms,_NMC,printlog)
        one_to_SNR_Linear+=1/10**(SNR_SingleLink_dB/10)
        
    return -10*log10(one_to_SNR_Linear)
#%%
def SNR_EGN(EGN_Accessories,NLI_terms=None,_NMC=10e6,printlog=False):
    
    LinkTupleList            = EGN_Accessories['LinkTupleList']
    LinkNumSpanList          = EGN_Accessories['LinkNumSpanList']
    AmplifierNoiseFigureList = EGN_Accessories['AmplifierNoiseFigureList']
    AmplifierGainList        = EGN_Accessories['AmplifierGainList']
    LPLaunchPowerDict        = EGN_Accessories['LPLaunchPowerDict']
    COILPLaunchPower_dBm     = EGN_Accessories['COILPLaunchPower_dBm']
    COILPSymbolRate          = EGN_Accessories['COILPSymbolRate']
#    COILPBandwidth           = EGN_Accessories['COILPBandwidth']
    
    NumLinks=len(LinkTupleList)
    
    NLI_terms_for_debugging=[]
    
    if NLI_terms==None:
        NLI_terms,NLI_terms_for_debugging,NLI_terms_for_GN=NLI_EGN(EGN_Accessories,_NMC,printlog)
    
    LPLaunchPowerDict_Linear={tup: 10**(0.1*LPLaunchPowerDict[tup]-3) for tup in LPLaunchPowerDict}
    
    NLI_var_coherent=0
#    NLI_var_link_level=0
    G_ASE=0
    
    for k in range(NumLinks):
        for kp in range(NumLinks):
            for tup1 in LinkTupleList[k] & LinkTupleList[kp]:
                for tupt in LinkTupleList[k] & LinkTupleList[kp]:
                    for tup2 in LinkTupleList[k] & LinkTupleList[kp]:
                        
                        if tup1[0]<tup2[0]:
                            continue
                        
                        NLI_var_coherent=NLI_var_coherent+\
                        NLI_terms[(k,kp,tup1,tupt,tup2)]*\
                        LPLaunchPowerDict_Linear[tup1]*\
                        LPLaunchPowerDict_Linear[tupt]*\
                        LPLaunchPowerDict_Linear[tup2]
#                        
#                        if k==kp:
#                            NLI_var_link_level=NLI_var_link_level+\
#                            NLI_terms[(k,kp,tup1,tupt,tup2)]*\
#                            LPLaunchPowerDict_Linear[tup1]*\
#                            LPLaunchPowerDict_Linear[tupt]*\
#                            LPLaunchPowerDict_Linear[tup2]
                            
    
    h=6.62607004e-34
    c=299792458
    C_lambda=1.55e-6
    nu=c/C_lambda
    
    for iLink in range(NumLinks):
        G_ASE=G_ASE+h*nu/2*(10**(AmplifierGainList[iLink][0]/10+AmplifierNoiseFigureList[iLink][0]/10)-1)*LinkNumSpanList[iLink]*2
        
    SNR_dB_coherent=COILPLaunchPower_dBm-30-10*log10(G_ASE*COILPSymbolRate+NLI_var_coherent)
#    SNR_dB_link_level=COILPLaunchPower_dBm-30-10*log10(G_ASE*COILPSymbolRate+NLI_var_link_level)
    
#    print()
    
    return SNR_dB_coherent
#    return SNR_dB_coherent,NLI_terms_for_GN#,SNR_dB_link_level,NLI_terms,NLI_terms_for_debugging#,10*log10(NLI_var_coherent),NLI_terms_for_debugging
#    return SNR_dB_coherent,SNR_dB_link_level,NLI_terms,NLI_terms_for_debugging#,10*log10(NLI_var_coherent),NLI_terms_for_debugging
#%%
def SNR_GN(EGN_Accessories,NLI_terms_for_GN=None,_NMC=10e6,printlog=False):
    
    LinkTupleList            = EGN_Accessories['LinkTupleList']
    LinkNumSpanList          = EGN_Accessories['LinkNumSpanList']
    AmplifierNoiseFigureList = EGN_Accessories['AmplifierNoiseFigureList']
    AmplifierGainList        = EGN_Accessories['AmplifierGainList']
    LPLaunchPowerDict        = EGN_Accessories['LPLaunchPowerDict']
    COILPLaunchPower_dBm     = EGN_Accessories['COILPLaunchPower_dBm']
    COILPSymbolRate          = EGN_Accessories['COILPSymbolRate']
#    COILPBandwidth           = EGN_Accessories['COILPBandwidth']
    
    NumLinks=len(LinkTupleList)
    
    NLI_terms_for_debugging=[]
    
    if NLI_terms_for_GN==None:
        NLI_terms,NLI_terms_for_debugging,NLI_terms_for_GN=NLI_EGN(EGN_Accessories,_NMC,printlog)
    
    LPLaunchPowerDict_Linear={tup: 10**(0.1*LPLaunchPowerDict[tup]-3) for tup in LPLaunchPowerDict}
    
    NLI_var_coherent=0
#    NLI_var_link_level=0
    G_ASE=0
    
    for k in range(NumLinks):
        for kp in range(NumLinks):
            for tup1 in LinkTupleList[k] & LinkTupleList[kp]:
                for tupt in LinkTupleList[k] & LinkTupleList[kp]:
                    for tup2 in LinkTupleList[k] & LinkTupleList[kp]:
                        
                        if tup1[0]<tup2[0]:
                            continue
                        
                        NLI_var_coherent=NLI_var_coherent+\
                        NLI_terms_for_GN[(k,kp,tup1,tupt,tup2)]*\
                        LPLaunchPowerDict_Linear[tup1]*\
                        LPLaunchPowerDict_Linear[tupt]*\
                        LPLaunchPowerDict_Linear[tup2]
#                        
#                        if k==kp:
#                            NLI_var_link_level=NLI_var_link_level+\
#                            NLI_terms[(k,kp,tup1,tupt,tup2)]*\
#                            LPLaunchPowerDict_Linear[tup1]*\
#                            LPLaunchPowerDict_Linear[tupt]*\
#                            LPLaunchPowerDict_Linear[tup2]
                            
    
    h=6.62607004e-34
    c=299792458
    C_lambda=1.55e-6
    nu=c/C_lambda
    
    for iLink in range(NumLinks):
        G_ASE=G_ASE+h*nu/2*(10**(AmplifierGainList[iLink][0]/10+AmplifierNoiseFigureList[iLink][0]/10)-1)*LinkNumSpanList[iLink]*2
        
    SNR_dB_coherent=COILPLaunchPower_dBm-30-10*log10(G_ASE*COILPSymbolRate+NLI_var_coherent)
#    SNR_dB_link_level=COILPLaunchPower_dBm-30-10*log10(G_ASE*COILPSymbolRate+NLI_var_link_level)
    
#    print()
    
    return SNR_dB_coherent#,SNR_dB_link_level,NLI_terms,NLI_terms_for_debugging#,10*log10(NLI_var_coherent),NLI_terms_for_debugging
#    return SNR_dB_coherent,SNR_dB_link_level,NLI_terms,NLI_terms_for_debugging#,10*log10(NLI_var_coherent),NLI_terms_for_debugging
#%%
def NLI_EGN(EGN_Accessories,_NMC=10000,printlog=False):
    
    LinkTupleList            = EGN_Accessories['LinkTupleList']
    LinkalphaList            = EGN_Accessories['LinkalphaList']
    Linkbeta2List            = EGN_Accessories['Linkbeta2List']
    LinkgammaList            = EGN_Accessories['LinkgammaList']
    LinkNumSpanList          = EGN_Accessories['LinkNumSpanList']
    LinkSpanLengthList       = EGN_Accessories['LinkSpanLengthList']
#    LinkNetGainList          = EGN_Accessories['LinkNetGainList']
    LinkAccDispersionList    = EGN_Accessories['LinkAccDispersion']
#    AmplifierNoiseFigureList = EGN_Accessories['AmplifierNoiseFigureList']
    ModPhiDict               = EGN_Accessories['ModPhiDict']
    ModPsiDict               = EGN_Accessories['ModPsiDict']
#    LPLaunchPowerDict        = EGN_Accessories['LPLaunchPowerDict']
#    LProll_off_factorDict    = EGN_Accessories['LProll_off_factorDict']
#    LPBandwidthDict          = EGN_Accessories['LPBandwidthDict']
#    LPSymbolRateDict         = EGN_Accessories['LPSymbolRateDict']
    LPNodeIDs                = EGN_Accessories['LinkNodeindList']
    COILPWavelength          = EGN_Accessories['COILPWavelength']
    COILPBandwidth           = EGN_Accessories['COILPBandwidth']
    
    NumLinks=len(LinkNumSpanList)
#    print('aa;sla;sl;als;als = ',ModPsiDict)
    
    TotalAccDispersionOverLinks=zeros([NumLinks+1,NumLinks+1])
    for iLink in range(NumLinks+1):
        for jLink in range(NumLinks+1):
            TotalAccDispersionOverLinks[iLink,jLink]=sum(LinkAccDispersionList[iLink:jLink])
    
#    freq_interval=uniform(COILPWavelength-COILPBandwidth/2,COILPWavelength+COILPBandwidth/2,20)
    
    NLI_terms={}
    NLI_terms_for_debugging={}
    NLI_terms_for_GN={}
    
#    for f in freq_interval:
#    tqdm(range(NumofFiberSections),position=0,leave=True):
    
#    for k in tqdm(range(NumLinks),position=0,leave=True):
    for k in range(NumLinks):
        
        Lspan_k=LinkSpanLengthList[k][0]
        alpha_k=LinkalphaList[k][0]
        beta2_k=Linkbeta2List[k][0]
        gamma_k=LinkgammaList[k][0]
        Nspan_k=LinkNumSpanList[k]
        
        for kp in range(NumLinks):
            
            Lspan_kp=LinkSpanLengthList[kp][0]
            alpha_kp=LinkalphaList[kp][0]
            beta2_kp=Linkbeta2List[kp][0]
            gamma_kp=LinkgammaList[kp][0]
            Nspan_kp=LinkNumSpanList[kp]
            
            TupleIntersection=LinkTupleList[k] & LinkTupleList[kp]
            
#            print('{},{} = {}\n\n'.format(k,kp,TupleIntersection))
            
#            if Lspan_k==Lspan_kp:
#                print('OK!')
#            if alpha_k==alpha_kp:
#                print('OK!')
#            if beta2_k==beta2_kp:
#                print('OK!')
#            if gamma_k==gamma_kp:
#                print('OK!')
#            if Nspan_k==Nspan_kp:
#                print('OK!')
            
            for tup1 in TupleIntersection:
                for tupt in TupleIntersection:
                    for tup2 in TupleIntersection:
                        
                        nu_kap1=tup1[0]
                        nu_kapt=tupt[0]
                        nu_kap2=tup2[0]
                        
#                        print(tup1)
#                        print(tupt)
#                        print(tup2)
                        
#                        print(nu_kap1)
#                        print(nu_kapt)
#                        print(nu_kap2)
                        
                        if nu_kap1<nu_kap2:
                            continue
                        
                        if printlog:
#                                print('............... Power = '+str(Power[(COI,SourceNode)])[:7])
                            print('     [k, kp] = ['+str(k)+', '+str(kp)+']')
                            print('[k1 ,kt ,k2] = ['+str(nu_kap1)+', '+str(nu_kapt)+', '+str(nu_kap2)+']\n')
#                                print('f = '+str(floor(f/BaudRate*100)/100))
                        
                        Omega=nu_kap1-nu_kapt+nu_kap2
                        
                        i1=LPNodeIDs.index(tup1[-1])
                        it=LPNodeIDs.index(tupt[-1])
                        i2=LPNodeIDs.index(tup2[-1])
                        
#                        print(i1)
#                        print(it)
#                        print(i2)
                        
                        '''SymbolRates'''
                        SR1=tup1[2]
                        SRt=tupt[2]
                        SR2=tup2[2]
                        '''Bandwidths'''
                        BW1=tup1[1]
#                        BWt=tupt[1]
                        BW2=tup2[1]
                        '''roll_off factors'''
                        roll_off_factor1=tup1[-2]
                        roll_off_factort=tupt[-2]
                        roll_off_factor2=tup2[-2]
                        
#                        '''SymbolRates'''
#                        SR1=LPSymbolRateDict[tup1]
#                        SRt=LPSymbolRateDict[tupt]
#                        SR2=LPSymbolRateDict[tup2]
#                        '''Bandwidths'''
#                        BW1=LPBandwidthDict[tup1]
#                        BWt=LPBandwidthDict[tupt]
#                        BW2=LPBandwidthDict[tup2]
#                        '''roll_off factors'''
#                        roll_off_factor1=LProll_off_factorDict[tup1]
#                        roll_off_factort=LProll_off_factorDict[tupt]
#                        roll_off_factor2=LProll_off_factorDict[tup2]
                        
#                        print(SR1)
#                        print(SRt)
#                        print(SR2)
                        
#                        print(BW1)
#                        print(BWt)
#                        print(BW2)
                        
                        D_temp=_3DMC(lambda f1,f2,f:
                            Upsilon(k,f1+nu_kap1,f2+nu_kap2,f,alpha_k,beta2_k,gamma_k,Lspan_k,Nspan_k,TotalAccDispersionOverLinks,i1,it,i2,NumLinks)*
                            conj(Upsilon(kp,f1+nu_kap1,f2+nu_kap2,f,alpha_kp,beta2_kp,gamma_kp,Lspan_kp,Nspan_kp,TotalAccDispersionOverLinks,i1,it,i2,NumLinks))*
                            abs(RRC_f(f1,SR1,roll_off_factor1))**2*
                            abs(RRC_f(f2,SR2,roll_off_factor2))**2*
                            abs(RRC_f(f1+f2-f+Omega,SRt,roll_off_factort))**2,
                            -BW1/2,BW1/2,
                            -BW2/2,BW2/2,
                            COILPWavelength-COILPBandwidth/2,COILPWavelength+COILPBandwidth/2
                            )
                        
                        D_temp=16/27*SR1*SRt*SR2*real(D_temp)
#                        D_temp=16/27*BW1*BWt*BW2*real(D_temp)
#                        D_temp=16/27*BW1*BWt*BW2*D_temp
                        
#                        print(COILPWavelength)
#                        print(COILPBandwidth)
                        
                        E_temp=F_temp=G_temp=0
                        
                        if tup2==tupt:
                            
                            E_temp=_4DMC(lambda f1,f2,f2p,f:
                                Upsilon(k,f1+nu_kap1,f2+nu_kap2,f,alpha_k,beta2_k,gamma_k,Lspan_k,Nspan_k,TotalAccDispersionOverLinks,i1,it,i2,NumLinks)*
                                conj(Upsilon(kp,f1+nu_kap1,f2p+nu_kap2,f,alpha_kp,beta2_kp,gamma_kp,Lspan_kp,Nspan_kp,TotalAccDispersionOverLinks,i1,it,i2,NumLinks))*
                                abs(RRC_f(f1,SR1,roll_off_factor1))**2*
                                RRC_f(f2,SR2,roll_off_factor2)*
                                conj(RRC_f(f1+f2-f+Omega,SR2,roll_off_factor2))*
                                conj(RRC_f(f2p,SR2,roll_off_factor2))*
                                RRC_f(f1+f2p-f+Omega,SR2,roll_off_factor2),
                                -BW1/2,BW1/2,
                                -BW2/2,BW2/2,
                                -BW2/2,BW2/2,
                                COILPWavelength-COILPBandwidth/2,COILPWavelength+COILPBandwidth/2
                                )
                            
#                            E_temp=80/81*BW1*BW2*ModPhiDict[tup2]*E_temp
                            E_temp=80/81*SR1*SR2*ModPhiDict[tup2]*real(E_temp)
#                            E_temp=80/81*BW1*BW2*ModPhiDict[tup2]*real(E_temp)
                        
                        if tup1==tup2:
                            
                            F_temp=_4DMC(lambda f1,f2,f1p,f:
                                Upsilon(k,f1+nu_kap1,f2+nu_kap2,f,alpha_k,beta2_k,gamma_k,Lspan_k,Nspan_k,TotalAccDispersionOverLinks,i1,it,i2,NumLinks)*
                                conj(Upsilon(kp,f1p+nu_kap1,f1+f2-f1p+nu_kap2,f,alpha_kp,beta2_kp,gamma_kp,Lspan_kp,Nspan_kp,TotalAccDispersionOverLinks,i1,it,i2,NumLinks))*
                                abs(RRC_f(f1+f2-f+Omega,SRt,roll_off_factort))**2*
                                RRC_f(f1,SR1,roll_off_factor1)*
                                conj(RRC_f(f1+f2-f1p,SR1,roll_off_factor1))*
                                conj(RRC_f(f1p,SR1,roll_off_factor1))*
                                RRC_f(f2,SR1,roll_off_factor1),
                                -BW1/2,BW1/2,
                                -BW2/2,BW2/2,
                                -BW1/2,BW1/2,
                                COILPWavelength-COILPBandwidth/2,COILPWavelength+COILPBandwidth/2
                                )
                            
#                            F_temp=16/81*BW1*BWt*ModPhiDict[tup1]*F_temp
                            F_temp=16/81*SR1*SRt*ModPhiDict[tup1]*real(F_temp)
#                            F_temp=16/81*BW1*BWt*ModPhiDict[tup1]*real(F_temp)
                        
                        if tup1==tup2==tupt:
                            
                            G_temp=_5DMC(lambda f1,f2,f1p,f2p,f:
                                Upsilon(k,f1+nu_kap1,f2+nu_kap2,f,alpha_k,beta2_k,gamma_k,Lspan_k,Nspan_k,TotalAccDispersionOverLinks,i1,it,i2,NumLinks)*
                                conj(Upsilon(kp,f1p+nu_kap1,f2p+nu_kap2,f,alpha_kp,beta2_kp,gamma_kp,Lspan_kp,Nspan_kp,TotalAccDispersionOverLinks,i1,it,i2,NumLinks))*
                                RRC_f(f1,SR1,roll_off_factor1)*
                                RRC_f(f2,SR1,roll_off_factor1)*
                                conj(RRC_f(f1p,SR1,roll_off_factor1))*
                                conj(RRC_f(f2p,SR1,roll_off_factor1))*
                                RRC_f(f1p+f2p-f+nu_kap1,SR1,roll_off_factor1)*
                                conj(RRC_f(f1+f2-f+nu_kap1,SR1,roll_off_factor1)),
                                -BW1/2,BW1/2,
                                -BW1/2,BW1/2,
                                -BW1/2,BW1/2,
                                -BW1/2,BW1/2,
                                COILPWavelength-COILPBandwidth/2,COILPWavelength+COILPBandwidth/2
                                )
                                
#                            G_temp=16/81*BW1*ModPsiDict[tup1]*G_temp
                            G_temp=16/81*SR1*ModPsiDict[tup1]*real(G_temp)
#                            G_temp=16/81*BW1*ModPsiDict[tup1]*real(G_temp)
                        
#                        E_temp=F_temp=G_temp=0
#                        print('=======',D_temp)
#                        print('===========',E_temp)
#                        print('===============',F_temp)
#                        print('===================',G_temp)
                        NLI_terms[(k,kp,tup1,tupt,tup2)]=(D_temp+E_temp+F_temp+G_temp)*((tup1!=tup2)+1)
                        
                        NLI_terms_for_GN[(k,kp,tup1,tupt,tup2)]=D_temp*((tup1!=tup2)+1)
                        
                        NLI_terms_for_debugging[
                                (k,kp,(tup1[0]/BW1,tup1[1]),(tupt[0]/BW1,tupt[1]),(tup2[0]/BW1,tup2[1]))
                                ]=(D_temp,E_temp,F_temp,G_temp)
#                        NLI_terms[(k,kp,tup1,tupt,tup2)]=(D_temp+E_temp+F_temp+G_temp)*((tup1!=tup2)+1)
    return NLI_terms,NLI_terms_for_debugging,NLI_terms_for_GN
#%%
'''Beginning of CFM'''
'''
[
 [Span1_Ch1,Span1_Ch2],
 [Span2_Ch1,Span2_Ch2]
]

e.g. G_bar[2][3] refers to *G_bar* at the 3rd span and the 4th channel

The upper index anywhere, denotes the number of rows or the length of the lists.
The lower index anywhere, denotes the number of columns or the length of the inner lists.

e.g. for f_comb:
    f_comb=[[1,3,5],[3,4,7,8]]
    
    The 1st span contains lambdas [1,3,5] and the 2nd span contains lambdas [3,4,7,8]

IMPORTANT:
    ***The alpha introduced in the following formulation is twice as much as that defined
    in the lecture.***
'''
#%%
def toarr(x):
    try:
        return array([array(ind) for ind in x])
    except TypeError:
        return x
#%%
def SuperSum(x,y):
    assert len(x)==len(y), 'Array lengths not equal'
    x=toarr(x)
    y=toarr(y)
    return array([array(x_comp)+array(y_comp) for (x_comp,y_comp) in zip(x,y)])
def SuperProd(x,y):
    assert len(x)==len(y), 'Array lengths not equal'
    x=toarr(x)
    y=toarr(y)
    return array([array(x_comp)*array(y_comp) for (x_comp,y_comp) in zip(x,y)])
def SuperDiv(x,y):
    assert len(x)==len(y), 'Array lengths not equal'
    x=toarr(x)
    y=toarr(y)
    return array([array(x_comp)/array(y_comp) for (x_comp,y_comp) in zip(x,y)])
def SuperSub(x,y):
    assert len(x)==len(y), 'Array lengths not equal'
    x=toarr(x)
    y=toarr(y)
    return array([array(x_comp)-array(y_comp) for (x_comp,y_comp) in zip(x,y)])
#def Gamma(f_CUT,NumSpan,alphaSpan_in_f_CUT,LengthSpan,iscompensated=True):
#%%
def HN(n):
    return sum(1/arange(1,n+1))
#%%
def Gamma_at_f_CUT(alphaspanvec_in_f_CUT,LengthSpanvec,iscompensated=True):
    '''
    The power-gain/loss at frequency *f_CUT* due to lumped
    elements, such as amplifiers and gain-flattening
    filters (GFFs), placed at the end of the span fiber...
    
    *alphaSpan_in_f_CUT* is a list of floats of length *NumSpan* with each of its entries
    representing the span attenuation at *f_CUT*
    
    *LengthSpanvec* is a list of floats of length *NumSpan* with each of its entries
    representing the span length
    
    Its return type is a list of floats of length *NumSpan*
    '''
    alphaspanvec_in_f_CUT=array(alphaspanvec_in_f_CUT)
    LengthSpanvec=array(LengthSpanvec)
    
    if iscompensated:
        return exp(alphaspanvec_in_f_CUT*LengthSpanvec)
    else:
        '''Return a list of floats of length *NumSpan*'''
        return
#%%
def alpha_CFM(f_comb,alphaspanvec):
    '''
    The span attenuation vector...
    
    *f_comb*: list of length **NumSpan**, each of its elements being
    a list containing set of wavelengths inside each span
    
    *alphaSpanvec* : list of floats of length *NumSpan*
    
    *alpha_CFM* gives the signal attenuation not power attenuation
    
    Its return type is just like *f_comb*
    '''
    f_comb=toarr(f_comb)
    alphaspanvec=array(alphaspanvec)
    return SuperSum(f_comb*0,alphaspanvec)
#%%
def beta2_bar(f_comb,f_CUT,beta2spanvec,beta3spanvec=None,f_c=None):
    '''
    Eq [5]
    
    The frequency f_c is where β_2 and β_3
    are calculated in the n-th span (float type)
    
    *f_CUT* : float
    
    *f_c*   : list of floats of length *NumSpan*
    
    *β_2*   : list of floats of length *NumSpan*
    
    *β_3*   : list of floats of length *NumSpan*
    
    *beta2spanvec* : list of floats of length *NumSpan*
    
    *beta3spanvec* : list of floats of length *NumSpan*
    
    *f_comb*: list of length **NumSpan**, each of its elements being
    a list containing set of wavelengths inside each span
    
    Its return type is just like *f_comb*
    '''
    f_comb=toarr(f_comb)
    
    if beta3spanvec==None:
        beta3spanvec=[0]*len(f_comb)
        
    if f_c==None:
        f_c=[0]*len(f_comb)
    
    beta2spanvec=array(beta2spanvec)
    beta3spanvec=array(beta3spanvec)
    
    f_c=array(f_c)
    return SuperSum(SuperProd(pi*beta3spanvec,SuperSum(f_comb+f_CUT,-2*f_c)),beta2spanvec)
#%%
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
A quantity *CUTindex_comb* is defined hereafter as a list of integers
of length *NumSpan* to contain CUT index at each span. It must be
used from now on.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#%%
def I(f_comb,R_comb,f_CUT,R_CUT,CUTindex_comb,alphaspanvec,beta2spanvec,Lengthspanvec,beta3spanvec,f_c,model_kind='CFM4'):
    '''
    I at each span
    Eqs [3,4]
    
    Its return type is just like *f_comb*
    '''
    f_comb=toarr(f_comb)
    R_comb=toarr(R_comb)
    
    alphaspanvec=array(alphaspanvec)
    beta2spanvec=array(beta2spanvec)
    beta3spanvec=array(beta3spanvec)
    Lengthspanvec=array(Lengthspanvec)
#    roll_off_factor_nch=array(roll_off_factor_nch)
    
#    print(beta3spanvec)
#    print(beta2spanvec)
    temp_beta2=beta2_bar(f_comb,f_CUT,beta2spanvec,beta3spanvec,f_c)
    temp_alpha=alpha_CFM(f_comb,alphaspanvec)
    
#    print(alphaspanvec)
#    print(beta2spanvec)
#    print(beta3spanvec)
#    print(Lengthspanvec)
    
    temp_arg1=pi**2*abs(temp_beta2/temp_alpha)*(f_comb-f_CUT+R_comb/2)*R_CUT
    temp_arg2=pi**2*abs(temp_beta2/temp_alpha)*(f_comb-f_CUT-R_comb/2)*R_CUT
    
#    print(temp_arg1)
    
#    print(beta2_bar(f_comb,f_CUT,beta2spanvec,beta3spanvec,f_c))
    
    temp1=array([arcsinh(tempind1) for tempind1 in temp_arg1])
    temp2=array([arcsinh(tempind2) for tempind2 in temp_arg2])
    
    full_term_I=(temp1-temp2)/(4*pi*abs(temp_beta2)*temp_alpha)
    
    if model_kind=='CFM1' or model_kind=='CFM2':
#        print(full_term_I)
        return full_term_I
    
    elif model_kind=='CFM3' or model_kind=='CFM4':
        
        Nspan=len(f_comb)
        
        for spanind in range(Nspan):
            beta2_at_CUT=temp_beta2[spanind][CUTindex_comb[spanind]]
            alpha_at_CUT=temp_alpha[spanind][CUTindex_comb[spanind]]
            Length_at_CUT=Lengthspanvec[spanind]
            
            InsideBracket=arcsinh(pi**2/2*abs(beta2_at_CUT/alpha_at_CUT)*R_CUT**2)+\
            4*sici(pi**2*abs(beta2_at_CUT)*Length_at_CUT*R_CUT**2)[0]/pi/alpha_at_CUT/\
            Length_at_CUT*(HN(Nspan)-1)
            
            I_CUT_scalar_temp=1/2/pi/abs(beta2_at_CUT)/alpha_at_CUT*InsideBracket
            
            full_term_I[spanind][CUTindex_comb[spanind]]=I_CUT_scalar_temp
        
#        print(full_term_I)
        return full_term_I
            
    else:
        raise Exception('CFM type not understood')
#%%
def G_bar(R_comb,Power_comb):
    '''Its return type is just like *f_comb*'''
    return toarr(Power_comb)/toarr(R_comb)
#%%
def beta2_bar_acc(f_comb,f_CUT,LengthSpanvec,beta2spanvec,beta3spanvec=None,f_c=None):
    
    '''
    IMPORTANT!!!!!!
    
    Temporary function for single link, multi-channel case
    
    Perhaps working for network-level simulation with high probability
    '''
    list_of_all_freqs=set()
    for f_comb_at_span in f_comb:
        list_of_all_freqs=list_of_all_freqs|set(f_comb_at_span)
    list_of_all_freqs=list(list_of_all_freqs)
    
    beta2_bar_acc_temp=array([cumsum(beta2spanvec*array(LengthSpanvec))]).T+[0]*len(list_of_all_freqs)
    beta2_bar_acc_temp=roll(beta2_bar_acc_temp,1,axis=0)
    beta2_bar_acc_temp[0]=0
    
    return beta2_bar_acc_temp,list_of_all_freqs
#%%
def rho(f_comb,f_CUT,R_CUT,LengthSpanvec,beta2spanvec,roll_off_factor_nch,beta3spanvec=None,f_c=None,Phi=0,model_kind='CFM1'):
    
    a1 = +1.0436e0; a13 = +1.0229e0
    a2 = -1.1878e0; a14 = -1.1440e0
    a3 = +1.0573e0; a15 = +1.1393e-2
    a4 = -1.8309e+1; a16 = +3.8070e+5
    a5 = +1.6665e0 ;a17 = +1.4785e+3
    a6 = -1.0020e0; a18 = -2.2593e0
    a7 = +9.0933e0; a19 = -6.7997e-1
    a8 = +6.6420e-3; a20 = +2.0215e0
    a9 = +8.4481e-1; a21 = -2.9781e-1
    a10 = -1.8530e0; a22 = +5.5130e-1
    a11 = +9.4539e-1; a23 = -3.6718e-1
    a12 = -1.5421e+1; a24 = +1.1486e0
    
    beta2_bar_acc_temp,list_of_all_freqs=beta2_bar_acc(f_comb,f_CUT,LengthSpanvec,beta2spanvec,beta3spanvec,f_c)
    
    temprho=[]
    
    r_CUT=roll_off_factor_nch[f_CUT]
    
    Phi={tempfreq__:1 for tempfreq__ in list_of_all_freqs}
    
    for spanind in range(len(f_comb)):
        InsiderhoList=[]
        for tempfreq in f_comb[spanind]:
            temp_r=roll_off_factor_nch[tempfreq]
            
            beta2_bar_acc_temp_scalar=beta2_bar_acc_temp[spanind][0]
            
            if tempfreq==f_CUT:
                InsiderhoList.append(
                        (1+a23*r_CUT**a24)*(
                                a9+a10*Phi[tempfreq]**a11+a12*Phi[tempfreq]**a13*\
                                (1+a14*R_CUT**a15+a16*(abs(beta2_bar_acc_temp_scalar)+a17)**a18))
                        )
            else:
                InsiderhoList.append(
                        (1+a19*r_CUT**a20+a21*temp_r**a22)*(
                                a1+a2*Phi[tempfreq]**a3+a4*Phi[tempfreq]**a5*\
                                (1+a6*(abs(beta2_bar_acc_temp_scalar)+a7)**a8))
                        )
        temprho.append(InsiderhoList)
    
#    print(temprho)
    
    return temprho
#%%
def G_NLI_Rx_f_CUT(f_comb,R_comb,f_CUT,R_CUT,Power_comb,alphaspanvec,beta2spanvec,roll_off_factor_nch,beta3spanvec,gammaspanvec,LengthSpanvec,f_c,model_kind='CFM4'):
    '''
    *gammaSpan*  :  the NLI coefficient
    
    *LengthSpan* :  the span length
    
    *GammaSpan*  :  the span amplifer power profile
    
    *alphaSpan*  :  the span attenuation coefficient
    
    *G_bar_n*    :  G_bar in Eq [6] at the n-th span
    
    *rho_n*      :  the rho in Eq [2] (len = NumChannelsperSpan)
    
    *I_n*        :  the I in Eq [2] (len = NumChannelsperSpan)
    
    "X" denoted built-in variables calculated from the corresponding functions.
    '''
    alphaspanvec=array(alphaspanvec)
    beta2spanvec=array(beta2spanvec)
    beta3spanvec=array(beta3spanvec)
    gammaspanvec=array(gammaspanvec)
#    print(gammaspanvec)
#    print(beta3spanvec)
#    print(beta2spanvec)
    
    Phi=0
    
    CUTindex_comb=array([list(spanfreq).index(f_CUT) for spanfreq in f_comb])
    
    XG_bar=G_bar(R_comb,Power_comb)
    Xrho=rho(f_comb,f_CUT,R_CUT,LengthSpanvec,beta2spanvec,roll_off_factor_nch,beta3spanvec,f_c,Phi,model_kind)
#    Xrho=rho(f_comb)
    XI=I(f_comb,R_comb,f_CUT,R_CUT,CUTindex_comb,alphaspanvec,beta2spanvec,LengthSpanvec,beta3spanvec,f_c,model_kind)
    
#    print('dlkfdslkfl;s\n',XI)
#    print(f_CUTindex_comb)
#    print(len(f_comb[0]))
    
    Xalpha_CFM=alpha_CFM(f_comb,alphaspanvec)
    alphaSpanvec_in_f_CUT=toarr([Xalpha_CFM[i][CUTindex_comb[i]] for i in range(len(Xalpha_CFM))])
    XG_bar_in_f_CUT=[XG_bar[i][CUTindex_comb[i]] for i in range(len(XG_bar))]
    XGamma=Gamma_at_f_CUT(alphaSpanvec_in_f_CUT,LengthSpanvec)
    
#    print(alphaSpanvec_in_f_CUT)
#    print(Xalpha_CFM)
#    print(XGamma)
#    print(Xrho)
#    print(XG_bar)
#    print(XI)
#    print(type(XGamma))
#    print(XG_bar_in_f_CUT)
#    return
    
    AmplifierGain_multiplied_by_Span_Attenuation=XGamma*exp(-alphaSpanvec_in_f_CUT*LengthSpanvec)
#    print(AmplifierGain_multiplied_by_Span_Attenuation)
    ### Term inside the parenthesis ###
    temp2_1=Xrho*XG_bar**2*XI
    temp2_2=temp2_1*array([2-(spanfreq==f_CUT) for spanfreq in f_comb]) # Gives 1 whenever f_comb=f_CUT otherwise gives 2
    temp2=array([sum(temp2_2ind) for temp2_2ind in temp2_2])
    ###################################
    temp1=16/27*gammaspanvec**2*AmplifierGain_multiplied_by_Span_Attenuation*XG_bar_in_f_CUT
    
#    print(temp1)
    
    G_NLI_vec=temp1*temp2
    
    G_NLI_Rx_f_CUT_temp=G_NLI_vec*prod(AmplifierGain_multiplied_by_Span_Attenuation)/cumprod(AmplifierGain_multiplied_by_Span_Attenuation)
    
#    print(sum(G_NLI_Rx_f_CUT_temp))
    
    return sum(G_NLI_Rx_f_CUT_temp)
#%%
#def ASE_PSD(NumSpan,AmplifierGain,AmplifierNF):
#    
#    h=6.62607004e-34
#    c=299792458
#    C_lambda=1.53e-6
#    nu=c/C_lambda
#    
#    AmplifierNoiseFactor=10**(AmplifierNF/10)    
#    return h*nu/2*NumSpan*(AmplifierNoiseFactor*AmplifierGain-1)
#%%
def SNR_CFM(CFM_Accessories,model_kind='CFM4'):
    f_comb                   = toarr(CFM_Accessories['f_comb'])
    R_comb                   = toarr(CFM_Accessories['R_comb'])
    Power_comb_dBm           = toarr(CFM_Accessories['Power_comb'])
    alphaspanvec             = toarr(CFM_Accessories['alpha'])
    beta2spanvec             = toarr(CFM_Accessories['beta2'])
    gammaspanvec             = toarr(CFM_Accessories['gamma'])
    alphaspanvec             = toarr(CFM_Accessories['alpha'])
    LengthSpanvec            = toarr(CFM_Accessories['Length'])
    AmplifierNoiseFigureList = toarr(CFM_Accessories['AmplifierNoiseFigureList'])
    AmplifierGain_dB_List    = toarr(CFM_Accessories['AmplifierGain'])
    roll_off_factor_nch      = CFM_Accessories['roll_off_factor_nch']
    f_CUT                    = CFM_Accessories['f_CUT']
    
    beta3spanvec=None
    f_c=None
    
#    print(Power_comb_dBm)
#    print(f_comb)
#    print(R_comb)
#    print(alphaspanvec)
#    print(beta2spanvec)
#    print(beta3spanvec)
#    print(gammaspanvec)
    
    Power_comb=[10**(0.1*array(powerspan)-3) for powerspan in Power_comb_dBm]
    
#    print(Power_comb)
    
    CUTindex_at_firstSpan=list(f_comb[0]).index(f_CUT)
    
    R_CUT=R_comb[0][CUTindex_at_firstSpan]
    Power_at_CUT=Power_comb[0][CUTindex_at_firstSpan]
    
#    print(Power_at_CUT)
    
    h=6.62607004e-34
    c=299792458
    C_lambda=1.55e-6
    nu=c/C_lambda
    
#    print(AmplifierGain)
#    print(AmplifierNoiseFactor)
    Total_NumSpan=len(f_comb)
#    print(Total_NumSpan)
    ASE_PSD=0
    for iSpan in range(Total_NumSpan):
        iAmpNoiseFactor=10**(AmplifierNoiseFigureList[iSpan]/10)
        iAmpGain=10**(AmplifierGain_dB_List[iSpan]/10)
        ASE_PSD=ASE_PSD+h*nu/2*(iAmpNoiseFactor*iAmpGain-1)*2
#    print('.......CFM = ',ASE_PSD*R_CUT)
    
    NLI_PSD=G_NLI_Rx_f_CUT(
            f_comb,
            R_comb,
            f_CUT,
            R_CUT,
            Power_comb,
            alphaspanvec,
            beta2spanvec,
            roll_off_factor_nch,
            beta3spanvec,
            gammaspanvec,
            LengthSpanvec,
            f_c,
            model_kind
            )
#    print(NLI_PSD)
    
    return 10*log10(Power_at_CUT/(NLI_PSD+ASE_PSD)/R_CUT),10*log10(NLI_PSD*R_CUT)

'''End of CFM'''
#%%
'''Beginning of SSFM'''
class Transceiver:
    
#    def __init__(self,TransceiverID,Wavelength,SymbolRate,ChannelBandwidth,isCUT=False):
    def __init__(self,TransceiverID,SymbolRate,ChannelBandwidth):
        self.ID                   = TransceiverID
        self.Wavelength           = None # Transceiver channel center frequency
        self.SymbolRate           = SymbolRate
        self.ChannelBandwidth     = ChannelBandwidth # Transceiver channel bandwidth
        self.TxModule             = self.Tx()
        self.RxModule             = self.Rx()
#        self.upsampling_factor    = 2 # fsampling is twice as much as symbol rate to cover (1+roll_off)*symbolrate
#        self.RRCSamplingFrequency = self.upsampling_factor*SymbolRate
        self.RRCSamplingFrequency = 2*SymbolRate
        self.RRC_trunclength      = 0 # Must be EVEN!
        self.assignedLPID         = None
        
        self.lenTransSignal=0
        
        self.TxModule.SymbolRate=SymbolRate
        self.RxModule.SymbolRate=SymbolRate
        
        self.TxModule.ChannelBandwidth=ChannelBandwidth
        self.RxModule.ChannelBandwidth=ChannelBandwidth
        
    def setTxParams(self,n_sym,roll_off_factor,mod_type):
        self.TxModule.mod_type          = mod_type
        self.TxModule.n_sym             = n_sym
        self.TxModule.roll_off_factor   = roll_off_factor
        self.TxModule.ModulationSymbols = []
        self.TxModule.Power_dBm         = -1000
        self.RxModule.roll_off_factor   = roll_off_factor
        
        # Temporary
        if roll_off_factor<0.1:
#            self.RRC_trunclength=2000
            self.RRC_trunclength=1000
        else:
#            self.RRC_trunclength=100
            self.RRC_trunclength=50
        
        if __ISREMOVEDTRANSIENT__==True:
            self.lenTransSignal=2*n_sym
#            self.lenTransSignal=n_sym*self.upsampling_factor
        else:
#            self.lenTransSignal=self.RRC_trunclength+n_sym*self.upsampling_factor-1
            self.lenTransSignal=2*self.RRC_trunclength+2*n_sym
#            self.lenTransSignal=self.RRC_trunclength+2*n_sym-2
#            self.lenTransSignal=self.RRC_trunclength+n_sym*self.upsampling_factor-2
        
#        t_temp,rrc_filter_temp=rrcosfilter(self.RRC_trunclength,roll_off_factor,1,2)
        _,rrc_filter_temp=rrcosfilter(self.RRC_trunclength*2+2,roll_off_factor,1,2)
        
        self.TxModule.rrc_filter=rrc_filter_temp[1:]
        self.RxModule.rrc_filter=rrc_filter_temp[1:]
    
    def TransSignal(self):
        
        Power=10**(self.TxModule.Power_dBm*0.1-3)
        
        '''
        Power is considered as dual polarized, so that each polarization share
        is half the power amount. The division is by 4.
        '''
            
        D4_mods=D4()
        
        if self.TxModule.mod_type in D4_mods.ModulationDict:
            mod_alphabet=D4_mods.ModulationDict[self.TxModule.mod_type]
            len_mod_alphabet=len(mod_alphabet)
            syms=randint(0,len_mod_alphabet,self.TxModule.n_sym)
            mod_power=D4_mods.ModPowerDict[self.TxModule.mod_type]
            self.TxModule.ModulationSymbols=mod_alphabet[syms].T*sqrt(Power/mod_power)
        
        elif self.TxModule.mod_type=='8QAM':
            mod_alphabet=[1+1j,1-1j,-1+1j,-1-1j,1,1j,-1,-1j]
            mod_power=2*average(abs(mod_alphabet)**2)
            self.TxModule.ModulationSymbols=choice(mod_alphabet,[2,self.TxModule.n_sym])+\
            choice(mod_alphabet,[2,self.TxModule.n_sym])*1j
            
        else:
            raise Exception('Unknown Modulation Format')
            
        ModulationSymbols_Resampled=kron(self.TxModule.ModulationSymbols,[1,0])
        
        baseband=upfirdn(self.TxModule.rrc_filter,ModulationSymbols_Resampled)
        
        _HALFTRUNCLENGTH=self.RRC_trunclength
#        _HALFTRUNCLENGTH=int(self.RRC_trunclength/2)
        
        if __ISREMOVEDTRANSIENT__==True:
            baseband=array([
                    baseband[0][_HALFTRUNCLENGTH:-_HALFTRUNCLENGTH],
                    baseband[1][_HALFTRUNCLENGTH:-_HALFTRUNCLENGTH]
                    ])
                                
#            baseband=array([
#                    baseband[0][_HALFTRUNCLENGTH:1-_HALFTRUNCLENGTH],
#                    baseband[1][_HALFTRUNCLENGTH:1-_HALFTRUNCLENGTH]
#                    ])
        
        return baseband
        
    class Tx:
        
        def __init__(self):
            '''Primary Parameters'''
            self.Wavelength        = None
            self.SymbolRate        = None
            self.ChannelBandwidth  = None
            '''Secondary Parameters'''
            self.n_sym             = None
            self.Power_dBm         = None
            self.mod_type          = None
            self.ModulationSymbols = []
            self.roll_off_factor   = None
            self.rrc_filter        = None
            
    class Rx:
        
        def __init__(self):
            '''Primary Parameters'''
            self.Wavelength       = None
            self.SymbolRate       = None
            self.ChannelBandwidth = None
            '''Secondary Parameters'''
            self.rrc_filter       = None
            self.roll_off_factor  = None
        
        def to_baseband(self,ReceivedSignal,time_D):
            return ReceivedSignal*exp(-2j*pi*self.Wavelength*time_D)
        
        def AnalogLowPassFilter(self,ReceivedSignal,fsampling):
            ''' Ideal '''
            passband_freq=self.ChannelBandwidth
#            passband_freq=min([
#                    self.ChannelBandwidth,
#                    self.SymbolRate*(1+self.roll_off_factor)*1.0001,
#                    self.SymbolRate*1.0001
#                    ])
            sig_length=len(ReceivedSignal[0])
#            print('************************************\n'*20)
#            print(self.SymbolRate)
#            print(self.roll_off_factor)
#            print('************************************\n'*20)
            N_ones=int(passband_freq/fsampling*sig_length/2)
            filt_freq=[1]*N_ones+[0]*(sig_length-2*N_ones)+[1]*N_ones
            
            output=ifft(fft(ReceivedSignal)*filt_freq)
            
            return output
        
#        def Downsample(self,ReceivedSignal,fsampling):
#            
#            dnsampling_ratio=int(fsampling/2/self.SymbolRate)
            
        def MF_DownSample(self,ReceivedSignal,dnsampling_ratio):
#            dnsampling_ratio=int(fsampling/2/self.SymbolRate)
            
            '''Constructing UpSampled Matched Filter'''
            N_FILT_RES=int(len(self.rrc_filter)/2-0.5)*dnsampling_ratio*2+2
            _,rrc_filt_res=rrcosfilter(N_FILT_RES,self.roll_off_factor,1,2*dnsampling_ratio)
            rrc_filt_res=rrc_filt_res[1:]
            
            output=upfirdn(rrc_filt_res,ReceivedSignal)
            if __ISREMOVEDTRANSIENT__:
                HALF_N_FILT_RES=int(N_FILT_RES/2)-1
                output=[
                    output[0][HALF_N_FILT_RES:-HALF_N_FILT_RES],
                    output[1][HALF_N_FILT_RES:-HALF_N_FILT_RES]
                    ]
            output=resample_poly(output,up=1,down=dnsampling_ratio,axis=1)
            
#            output=resample_poly(ReceivedSignal.T,up=1,down=dnsampling_ratio).T
#            output=upfirdn(self.rrc_filter,output)
#            if __ISREMOVEDTRANSIENT__==True:
#                RRCFilterLength=len(self.rrc_filter)
#                HalfRRCFilterLength=int(RRCFilterLength/2)
#                output=array([
#                        output[0][HalfRRCFilterLength:-HalfRRCFilterLength],
#                        output[1][HalfRRCFilterLength:-HalfRRCFilterLength]
#                        ])
##                output=array([
##                        output[0][HalfRRCFilterLength:1-HalfRRCFilterLength],
##                        output[1][HalfRRCFilterLength:1-HalfRRCFilterLength]
##                        ])
#            else:
#                output=array([
#                        output[0][RRCFilterLength:-RRCFilterLength],
#                        output[1][RRCFilterLength:-RRCFilterLength]
#                        ])
#                output=array([
#                        output[0][RRCFilterLength:1-RRCFilterLength],
#                        output[1][RRCFilterLength:1-RRCFilterLength]
#                        ])
            
            return output#upfirdn(rrc_filter,yyy,down=int(fsampling/self.SymbolRate),up=1)
        
        def EDC(self,ReceivedSignal,AccDispersion,freq_D=[]):
            '''
            If freq_D is None, the EDC compensates for single channel
            , unless it compensates for all the C-band
            '''
            if list(freq_D)==[]:
                spacing=arange(len(ReceivedSignal[0]))/len(ReceivedSignal[0])
                freq_D_hat=spacing-floor(spacing+0.5)
    #            freq_D_at_CurrentLink=
                freq_D_at_EDC = freq_D_hat*2*self.SymbolRate
                return fft(ifft(ReceivedSignal)*exp(-2j*pi**2*freq_D_at_EDC**2*AccDispersion))
            else:
                return fft(ifft(ReceivedSignal)*exp(-2j*pi**2*freq_D**2*AccDispersion))
        
        def SymbolDetector(self,ReceivedSignal,TotalNumOfSymbols):
            '''Extracting received symbols from the received signal'''
            # Regularly picking symbols alternatively 
            Detected_ModulationSymbols=array([
                    ReceivedSignal[0][:(TotalNumOfSymbols-1)*2+1:2],
                    ReceivedSignal[1][:(TotalNumOfSymbols-1)*2+1:2]
                    ])#/sum(self.rrc_filter**2)
#            # Regularly picking symbols alternatively 
            
            return Detected_ModulationSymbols
        
#        def DecisionMaker(self,DetectedModulationSymbols):
#            '''
#            This function is not used in the main code, however it may come
#            handy later....
#            '''
#            power_linear=10**(0.1*self.Power_dBm-3)
#            
#            rx_real=real(DetectedModulationSymbols)
#            rx_imag=imag(DetectedModulationSymbols)
#            
#            det_mod_syms_real=(rx_real>0)*sqrt(power_linear/4)+(rx_real<0)*(-sqrt(power_linear/4))
#            det_mod_syms_imag=(rx_imag>0)*sqrt(power_linear/4)+(rx_imag<0)*(-sqrt(power_linear/4))
#            
#            det_mod_syms=det_mod_syms_real+1j*det_mod_syms_imag
#            
#            return det_mod_syms
        
        def PhaseRotationCompensator(self,RxModulationSymbols,TxModulationSymbols):
            '''
            The optimum phase rotation angle is that calculated through xy* where
            "x" denotes transmitted and "y" denoted received.
            '''
#            print('aksalkslaksl=',DetectedModulationSymbols)
#            print('aksalkslaksljskajskajs=',ModulationSymbols)
            opt_angle=array([
                    angle(sum(TxModulationSymbols*conj(RxModulationSymbols),1))
                    ]).T
    
#            return RxModulationSymbols*exp(1j*opt_angle),opt_angle
            return RxModulationSymbols*exp(1j*opt_angle)
#%%
class Node:
    
    def __init__(self,NodeID):
        self.ID                     = NodeID
        self.TransceiverDict        = {}
        self.NumofTransceivers      = 0
        self.TraversedLightPathDict = {} # Whether added/dropped or passed through, is recorded
        
    def addNewTransceiver(self,TransceiverID,SymbolRate,ChannelBandwidth):
#    def addNewTransceiver(self,TransceiverID,Wavelength,SymbolRate,ChannelBandwidth):
        self.TransceiverDict[TransceiverID]  = Transceiver(TransceiverID,SymbolRate,ChannelBandwidth)
#        self.TransceiverDict[TransceiverID]  = Transceiver(TransceiverID,Wavelength,SymbolRate,ChannelBandwidth)
        self.NumofTransceivers              += 1
    
    def Total_addedSignal(self,CUTSymbolRate,SetofActiveTransceiverIDs,UpSamplingRatio,lenCUTSignal):
        '''
        *lenCUTSignal* is the reference length that every node should consider. It is defined
        as the length of the signal *after upsampling is done inside the node*.
        '''
        fsampling=UpSamplingRatio*CUTSymbolRate
        
        time_D=arange(lenCUTSignal)/fsampling
        
        temp_Total_addedSignal=0
        
        for transceiverid in SetofActiveTransceiverIDs:
            
            transceiver=self.TransceiverDict[transceiverid]
            
            transceiver.TxModule.n_sym=int(ceil(lenCUTSignal/UpSamplingRatio))
            
            temp_transceiverSignal=transceiver.TransSignal()
            
            dnsampling_ratio=1000
            upsampling_ratio=ceil(dnsampling_ratio*fsampling/transceiver.RRCSamplingFrequency)
            
            temp_transceiverSignal_resampled=resample_poly(temp_transceiverSignal.T,up=upsampling_ratio,down=dnsampling_ratio).T
            
            '''Symbol removal only from the beginning'''
            temp_transceiverSignal_resampled=array([
                    temp_transceiverSignal_resampled[0][len(temp_transceiverSignal_resampled[0])-lenCUTSignal:],
                    temp_transceiverSignal_resampled[1][len(temp_transceiverSignal_resampled[0])-lenCUTSignal:]
                    ])
            
            temp_Total_addedSignal=temp_Total_addedSignal+temp_transceiverSignal_resampled*exp(2j*pi*transceiver.Wavelength*time_D)
        
        return temp_Total_addedSignal
#%%
class Link:
    
    '''
    Unidirectional Link
    
    Link ID = (InNode,OutNode)
    '''
    
    def __init__(self,InNode=None,OutNode=None):
        self.ID                        = (InNode,OutNode)
        self.SpanIDList                = []
        self.SpanList                  = []
        self.alphaList                 = []
        self.beta2List                 = []
        self.gammaList                 = []
        self.LengthList                = []
        self.AmplifierGainList         = []
        self.AmplifierNoiseFigureList  = []
        self.AccDispersion             = 0
        self.NetGain                   = 1
        self.NumSpan                   = 0
        self.TupleDict                 = {}
        self.LaunchPowerDict           = {}
        self.LightPathBandwidthDict    = {}
#        self.TotalOccupiedWDMBandWidth = 0
        
    def addSpan(self,SpanID,Length,alpha,beta2,gamma,AmplifierNoiseFigure=-1000,AmplifierGain_dB=None):
        self.SpanIDList               .append(SpanID)
        self.SpanList                 .append(self.Span(SpanID,Length,alpha,beta2,gamma,AmplifierNoiseFigure,AmplifierGain_dB))
        self.alphaList                .append(alpha)
        self.beta2List                .append(beta2)
        self.gammaList                .append(gamma)
        self.LengthList               .append(Length)
        self.AmplifierGainList        .append(self.SpanList[-1].AmplifierGain_dB)
        self.AmplifierNoiseFigureList .append(AmplifierNoiseFigure)
        self.NumSpan                  += 1
        self.AccDispersion            += beta2*Length
        self.NetGain                  *= exp(-alpha*Length)*10**(self.SpanList[-1].AmplifierGain_dB/10)
        
    def NLSE_Solver_Link(self,inputSignal_t,freq_D,ReceiverBandWidth,UpsamplingRatio,LenStep):
#    def NLSE_Solver_Link(self,inputSignal_t,freq_D,Power_dBm,SymbolRate,ReceiverBandWidth,UpsamplingRatio):
        '''
        NLSE Solver in Link
        
        "inputSignal_t" is the time-domain Electrical Field, which will
        be returned as output after affected by impairments
        
        "freq_D" is the discrete frequency generated alongside the signal
        '''
        Exact_ASEvarperSpan_vec=[]
        Signal_ASENLIvarperSpan_vec=[]
        
        inputSignal_t_plus_NLI_ASE=inputSignal_t
        
        for spanind in range(self.NumSpan):
            print('......... span {} of {}'.format(spanind+1,self.NumSpan))
            Signal_ASENLIvarperSpan=sum(var(inputSignal_t_plus_NLI_ASE,1))
            inputSignal_f_plus_NLI,Full_Linear_Exp=self.SpanList[spanind].NLSE_Solver_Span(
                    inputSignal_t_plus_NLI_ASE,
                    freq_D,
                    LenStep
                    )
            
            '''Adding ASE noise to Signal'''
            AmplifierNoiseFactor=10**(self.SpanList[spanind].AmplifierNoiseFigure/10)
            
            AmplifierGain=10**(self.SpanList[spanind].AmplifierGain_dB/20)
            
            h=6.62607004e-34
            c=299792458
            C_lambda=1.55e-6
            nu=c/C_lambda
            
            '''ASE Variance based on exact mathematical formula'''
            N=len(inputSignal_f_plus_NLI[0])
            ASE_Variance=max([h*nu*(AmplifierNoiseFactor*AmplifierGain**2-1)*ReceiverBandWidth/2,0])
#            print('ASE',ASE_Variance*2)
            ASE_Variance=ASE_Variance*UpsamplingRatio
            
            noise_process=sqrt(ASE_Variance/2)*randn(2,N)+1j*sqrt(ASE_Variance/2)*randn(2,N)
            
            inputSignal_t_plus_NLI_ASE=AmplifierGain*fft(inputSignal_f_plus_NLI)+noise_process
            
            Exact_ASEvarperSpan_vec.append(ASE_Variance)
            
            Signal_ASENLIvarperSpan_vec.append(Signal_ASENLIvarperSpan)
            
        return inputSignal_t_plus_NLI_ASE,Exact_ASEvarperSpan_vec,Signal_ASENLIvarperSpan_vec,Full_Linear_Exp
        
    class Span:
    
        def __init__(self,SpanID,Length,alpha,beta2,gamma,AmplifierNoiseFigure=-1000,AmplifierGain_dB=None):
            self.SpanID=SpanID
            self.alpha=alpha
            self.beta2=beta2
            self.gamma=gamma
            self.Length=Length
            self.AmplifierNoiseFigure=AmplifierNoiseFigure
            if AmplifierGain_dB==None:
                self.AmplifierGain_dB=10*log10(exp(alpha*Length))
            else:
                self.AmplifierGain_dB=AmplifierGain_dB
    
        def NLSE_Solver_Span(self,inputSignal_t,freq_D,LenStep):
            '''
            NLSE Solver in Span
            
            "inputSignal_f" is the frequency-domain Electrical Field, which will
            be returned as output after affected by impairments
            
            "freq_D" is the discrete frequency generated alongside the signal
            '''
            alpha=self.alpha
            beta2=self.beta2
            gamma=self.gamma
            Length=self.Length
            
            if beta2!=0 and gamma!=0:
                NumofFiberSections=int(ceil(self.Length/LenStep))
            else:
                NumofFiberSections=1
            
            h=Length/NumofFiberSections
            
            inputSignal_f=ifft(inputSignal_t)
            
            # Linear Impairments, Full and half section
            Full_Linear_Exp=exp(-alpha*h/2+2j*beta2*h*pi**2*freq_D**2)
            Half_Linear_Exp=exp(-alpha*h/4+1j*beta2*h*pi**2*freq_D**2)
            
            ''' First half step '''
            inputSignal_f_1=inputSignal_f*Half_Linear_Exp
            
            if alpha==0:
                h_eff=h
            else:
                h_eff=(1-exp(-alpha*h))/alpha
            
            ''' Solver core '''
            for i in tqdm(range(NumofFiberSections),position=0,leave=True):
                ''' At each step: First Nonlinear, then linear '''
                inputSignal_f_1_inv=fft(inputSignal_f_1)
                inputSignal_f_2=inputSignal_f_1_inv*exp(1j*h_eff*gamma*sum(abs(inputSignal_f_1_inv)**2,0)*8/9)
                inputSignal_f_1=ifft(inputSignal_f_2)*Full_Linear_Exp
                
            ''' Half step linear backwards '''
            inputSignal_f_plus_NLI=inputSignal_f_1/Half_Linear_Exp
            
            return inputSignal_f_plus_NLI,Full_Linear_Exp
#%%
class LightPath:
    
    def __init__(self,LPID):
        self.ID                 = LPID
        self.LaunchPower        = None
        self.Wavelength         = None
        self.LightPathBandwidth = None
        self.SymbolRate         = None
        self.ModulationType     = None
        self.SrcTransceiverID   = None
        self.DestTransceiverID  = None
        self.roll_off_factor    = None
        self.NodeList           = []
        self.LinkList           = []
#%%
class Network:
    
    def __init__(self):
        self.NodeDict       = {}
        self.LinkDict       = {}
        self.NumofNodes     = 0
        self.NumofLinks     = 0
        self.LightPathDict  = {}
    
    def addNode(self,NodeID):
        self.NodeDict[NodeID] = Node(NodeID)
        self.NumofNodes      += 1
    
    def addLink(self,InNode,OutNode):
        self.LinkDict[(InNode,OutNode)] = Link(InNode,OutNode)
        self.NumofLinks                += 1
    
    def addLightPath(self,LPID,NodeList,Wavelength,LaunchPower):
#    def addLightPath(self,LPID,NodeList,SrcTransceiverID,DestTransceiverID,LaunchPower):
        tempLightPath                    = LightPath(LPID)
        
        SrcNodeID                        = NodeList[0]
        DestNodeID                       = NodeList[-1]
        
        SrcTransceiverID=DestTransceiverID=None
        
        for iTransceiver in self.NodeDict[SrcNodeID].TransceiverDict.values():
            if iTransceiver.assignedLPID==None:
                iTransceiver.assignedLPID        = LPID
                iTransceiver.Wavelength          = Wavelength
                iTransceiver.TxModule.Wavelength = Wavelength
                iTransceiver.RxModule.Wavelength = Wavelength
                SrcTransceiverID                 = iTransceiver.ID
                break
        
        for iTransceiver in self.NodeDict[DestNodeID].TransceiverDict.values():
            if iTransceiver.assignedLPID==None:
                iTransceiver.assignedLPID        = LPID
                iTransceiver.Wavelength          = Wavelength
                iTransceiver.TxModule.Wavelength = Wavelength
                iTransceiver.RxModule.Wavelength = Wavelength
                DestTransceiverID                = iTransceiver.ID
#                print('sld;dslf;sdlf;     ',DestTransceiverID)
                break
        
        if SrcTransceiverID==None or DestTransceiverID==None:
            raise Exception('Lightpath {} could not be established due to lack of transceivers!'.format(LPID))
        
        tempLightPath.SrcTransceiverID   = SrcTransceiverID
        tempLightPath.DestTransceiverID  = DestTransceiverID
        
        tempLightPath.LaunchPower        = LaunchPower
        
        tempLightPath.NodeList           = NodeList
        LinkList                         = list(zip(NodeList,NodeList[1:]))
        tempLightPath.LinkList           = LinkList
        
#        tempLightPath.Wavelength         = self.NodeDict[SrcNodeID].TransceiverDict[SrcTransceiverID].Wavelength
        tempLightPath.Wavelength         = Wavelength
        tempLightPath.SymbolRate         = self.NodeDict[SrcNodeID].TransceiverDict[SrcTransceiverID].SymbolRate
        tempLightPath.LightPathBandwidth = self.NodeDict[SrcNodeID].TransceiverDict[SrcTransceiverID].ChannelBandwidth
        tempLightPath.ModulationType     = self.NodeDict[SrcNodeID].TransceiverDict[SrcTransceiverID].TxModule.mod_type
        tempLightPath.roll_off_factor    = self.NodeDict[SrcNodeID].TransceiverDict[SrcTransceiverID].TxModule.roll_off_factor
        
        self.NodeDict[SrcNodeID].TransceiverDict[SrcTransceiverID].TxModule.Power_dBm=LaunchPower
        
        for iLink in LinkList:
            # Updating links in the network
            self.LinkDict[iLink].TupleDict[LPID] = (
                    tempLightPath.Wavelength,
                    tempLightPath.LightPathBandwidth,
                    tempLightPath.SymbolRate,
#                    LaunchPower,
                    tempLightPath.roll_off_factor,
                    SrcNodeID
                    )
            
#            self.LinkDict[iLink].TupleDict[LPID] = {
#                    'Wavelength': tempLightPath.Wavelength,
#                    'Bandwidth': tempLightPath.LightPathBandwidth,
#                    'SymbolRate': tempLightPath.SymbolRate,
#                    'LaunchPower': LaunchPower,
#                    'roll_off_factor': tempLightPath.roll_off_factor,
#                    'SrcNodeID': SrcNodeID
#                    }
            
            self.LinkDict[iLink].LaunchPowerDict[LPID]        = LaunchPower
            self.LinkDict[iLink].LightPathBandwidthDict[LPID] = tempLightPath.LightPathBandwidth
        
        for iNode in range(len(NodeList)):
            if iNode==0:
                self.NodeDict[NodeList[iNode]].TraversedLightPathDict[LPID]=('add',LinkList[iNode],tempLightPath.Wavelength)
            elif iNode==len(NodeList)-1:
                self.NodeDict[NodeList[iNode]].TraversedLightPathDict[LPID]=(LinkList[iNode-1],'drop',tempLightPath.Wavelength)
            else:
                self.NodeDict[NodeList[iNode]].TraversedLightPathDict[LPID]=(LinkList[iNode-1],LinkList[iNode],tempLightPath.Wavelength)
                
        self.LightPathDict[LPID] = tempLightPath
        
    def delLightPath(self,LPID='all'):
        
        if LPID=='all':
            self.LightPathDict = {}
            
            for iNode in self.NodeDict.values():
                iNode.TraversedLightPathDict={}
                for iTransceiver in iNode.TransceiverDict.values():
                    if iTransceiver.assignedLPID!=None:
                        iTransceiver.assignedLPID        = None
                        iTransceiver.Wavelength          = None
                        iTransceiver.TxModule.Wavelength = None
                        iTransceiver.RxModule.Wavelength = None
                
                for iLink in self.LinkDict.values():
                    iLink.TupleDict={}
                    iLink.LaunchPowerDict={}
                    iLink.LightPathBandwidthDict={}
        else:
            raise Exception('Network delLightPath function not complete yet!')
                
    def Export(self,LPID,extype):
        
        if extype=='EGN':
            ''' It is assumed that the links consist equal spans '''
            CUTLightPath                                 = self.LightPathDict[LPID]
            CUTLinkList                                  = CUTLightPath.LinkList
            EGN_Accessories                              = {}
            EGN_Accessories['LinkNodeindList']           = CUTLightPath.NodeList
            EGN_Accessories['LinkTupleList']             = []
            EGN_Accessories['LinkalphaList']             = []
            EGN_Accessories['Linkbeta2List']             = []
            EGN_Accessories['LinkgammaList']             = []
            EGN_Accessories['LinkNumSpanList']           = []
            EGN_Accessories['LinkSpanLengthList']        = []
            EGN_Accessories['LinkNetGainList']           = []
            EGN_Accessories['LinkAccDispersion']         = []
            EGN_Accessories['AmplifierNoiseFigureList']  = []
            EGN_Accessories['AmplifierGainList']         = []
            EGN_Accessories['ModPhiDict']                = {}
            EGN_Accessories['ModPsiDict']                = {}
            EGN_Accessories['LPLaunchPowerDict']         = {}
#            EGN_Accessories['LPBandwidthDict']           = {}
#            EGN_Accessories['LPSymbolRateDict']          = {}
#            EGN_Accessories['LProll_off_factorDict']     = {}
            EGN_Accessories['COILPWavelength']           = CUTLightPath.Wavelength
            EGN_Accessories['COILPBandwidth']            = CUTLightPath.LightPathBandwidth
            EGN_Accessories['COILPSymbolRate']           = CUTLightPath.SymbolRate
            EGN_Accessories['COILPLaunchPower_dBm']      = CUTLightPath.LaunchPower
            
#            DoubleLinkList=list(zip())
            
            for CurrentLinkID,PreviousLinkID in zip(CUTLinkList,[None]+CUTLinkList):
                CurrentLink=self.LinkDict[CurrentLinkID]
                LPIDSetofCurrentLink=CurrentLink.TupleDict.keys()
#                try:
#                    PreviousLink=self.LinkDict[PreviousLinkID]
#                    LPIDSetofPreviousLink=PreviousLink.TupleDict.keys()
#                except AttributeError:
#                    LPIDSetofPreviousLink=set()
#                EGN_Accessories['LinkTupleList']           .append(set(iLink.TupleDict.values()))
                EGN_Accessories['LinkAccDispersion']       .append(CurrentLink.AccDispersion)
                EGN_Accessories['LinkalphaList']           .append(CurrentLink.alphaList)
                EGN_Accessories['Linkbeta2List']           .append(CurrentLink.beta2List)
                EGN_Accessories['LinkgammaList']           .append(CurrentLink.gammaList)
                EGN_Accessories['LinkNumSpanList']         .append(CurrentLink.NumSpan)
                EGN_Accessories['LinkSpanLengthList']      .append(CurrentLink.LengthList)
                EGN_Accessories['AmplifierGainList']       .append(CurrentLink.AmplifierGainList)
                EGN_Accessories['LinkNetGainList']         .append(CurrentLink.NetGain)
                EGN_Accessories['AmplifierNoiseFigureList'].append(CurrentLink.AmplifierNoiseFigureList)
                
                tempTupleSet=set()
                
                for iLightPathID in LPIDSetofCurrentLink:
                    iLightPath=self.LightPathDict[iLightPathID]
                    
                    tempLightPathLaunchPower=iLightPath.LaunchPower # dBm
                    
                    for iLinkID in iLightPath.LinkList:
                        
                        if iLinkID in CUTLightPath.LinkList:
                            NewSrcNodeID=iLinkID[0]
                            break
                        else:
                            tempLightPathLaunchPower+=10*log10(self.LinkDict[iLinkID].NetGain)
                    
                    NewLPTuple=(
                            iLightPath.Wavelength,
                            iLightPath.LightPathBandwidth,
                            iLightPath.SymbolRate,
#                            tempLightPathLaunchPower,
                            iLightPath.roll_off_factor,
                            NewSrcNodeID
                            )
                    
                    if iLightPath.ModulationType=='cube4_16':
                        EGN_Accessories['ModPhiDict'][NewLPTuple]=-1
                        EGN_Accessories['ModPsiDict'][NewLPTuple]=4
                    elif iLightPath.ModulationType=='QAM4_256':
                        EGN_Accessories['ModPhiDict'][NewLPTuple]=-.68
                        EGN_Accessories['ModPsiDict'][NewLPTuple]=2.08
                    else:
                        raise Exception('Code not complete for other types of modulations!')
                    
                    EGN_Accessories['LPLaunchPowerDict'][NewLPTuple]=tempLightPathLaunchPower
                    
                    tempTupleSet.add((
                            iLightPath.Wavelength,
                            iLightPath.LightPathBandwidth,
                            iLightPath.SymbolRate,
#                            tempLightPathLaunchPower,
                            iLightPath.roll_off_factor,
                            NewSrcNodeID
                            ))
                
                EGN_Accessories['LinkTupleList'].append(tempTupleSet)    
                
            return EGN_Accessories
        
        elif extype=='CFM':
            ''' *LinkDict* in this return type contains lists of link specs along a
            specified lightpath '''
            tempLPA=self.LightPathDict[LPID]
            CFM_Accessories                             = {}
            CFM_Accessories['Length']                   = []
            CFM_Accessories['alpha']                    = []
            CFM_Accessories['beta2']                    = []
            CFM_Accessories['gamma']                    = []
            CFM_Accessories['AmplifierGain']            = []
            CFM_Accessories['AmplifierNoiseFigureList'] = []
            CFM_Accessories['f_comb']                   = []
            CFM_Accessories['R_comb']                   = []
            CFM_Accessories['Power_comb']               = []
            CFM_Accessories['roll_off_factor_nch']      = {}
            CFM_Accessories['f_CUT']                    = tempLPA.Wavelength
#            Set_of_All_LPs=set()
#            List_of_All_lambdas=[]
#            for iLink in tempLPA.LinkList:
#                Set_of_All_LPs=Set_of_All_LPs|set(self.LinkDict[iLink].TupleDict.keys())
##                print(set(self.LinkDict[iLink].TupleDict.keys()))
##                print(self.LinkDict[iLink].TupleDict.keys())
##            print(Set_of_All_LPs)
#            for iLightPath in Set_of_All_LPs:
#                List_of_All_lambdas.append(self.LightPathDict[iLightPath].Wavelength)
                
#            print(List_of_All_lambdas)
            for iLink in tempLPA.LinkList:
                tempLink=self.LinkDict[iLink]
                CFM_Accessories['Length']                  .extend(tempLink.LengthList)
                CFM_Accessories['alpha']                   .extend(tempLink.alphaList)
                CFM_Accessories['beta2']                   .extend(tempLink.beta2List)
                CFM_Accessories['gamma']                   .extend(tempLink.gammaList)
                CFM_Accessories['AmplifierGain']           .extend(tempLink.AmplifierGainList)
                CFM_Accessories['AmplifierNoiseFigureList'].extend(tempLink.AmplifierNoiseFigureList)
                CFM_Accessories['f_comb']                  .extend([[Tuple[0] for Tuple in list(tempLink.TupleDict.values())]]*tempLink.NumSpan)
                CFM_Accessories['R_comb']                  .extend([list(tempLink.LightPathBandwidthDict.values())]*tempLink.NumSpan)
                CFM_Accessories['Power_comb']              .extend([list(tempLink.LaunchPowerDict.values())]*tempLink.NumSpan)
            for tempLP in self.LightPathDict.values():
                CFM_Accessories['roll_off_factor_nch'][tempLP.Wavelength]=tempLP.roll_off_factor
            return CFM_Accessories
        
#        elif extype=='SSFM':
#            return {
#                    'NetworkLPCloud' : self.LightPathCloud,
#                    'LPID'           : LPID
#                    }
        elif extype=='EGN_noncoh':
            '''Fiber attenuations must be fully compensated'''
            
            CUTLightPath=self.LightPathDict[LPID]
            
            EGN_noncoh_Accessories=[]
            
            for iLinkID in CUTLightPath.LinkList:
                iLink=self.LinkDict[iLinkID]
                temp_EGN_noncoh_Accessories={}
                temp_EGN_noncoh_Accessories['LinkNodeindList']=[0,1]
                temp_EGN_noncoh_Accessories['LinkTupleList']=[]
                
                temp_EGN_noncoh_Accessories['LinkalphaList']=[iLink.alphaList]
                temp_EGN_noncoh_Accessories['Linkbeta2List']=[iLink.beta2List]
                temp_EGN_noncoh_Accessories['LinkgammaList']=[iLink.gammaList]
                temp_EGN_noncoh_Accessories['LinkNumSpanList']=[iLink.NumSpan]
                temp_EGN_noncoh_Accessories['LinkSpanLengthList']=[iLink.LengthList]
                temp_EGN_noncoh_Accessories['LinkNetGainList']=[iLink.NetGain]
                temp_EGN_noncoh_Accessories['LinkAccDispersion']=[iLink.AccDispersion]
                temp_EGN_noncoh_Accessories['AmplifierNoiseFigureList']=[iLink.AmplifierNoiseFigureList]
                temp_EGN_noncoh_Accessories['AmplifierGainList']=[iLink.AmplifierGainList]
                temp_EGN_noncoh_Accessories['ModPhiDict']={}
                temp_EGN_noncoh_Accessories['ModPsiDict']={}
                temp_EGN_noncoh_Accessories['LPLaunchPowerDict']={}
                temp_EGN_noncoh_Accessories['COILPWavelength']=CUTLightPath.Wavelength
                temp_EGN_noncoh_Accessories['COILPBandwidth']=CUTLightPath.LightPathBandwidth
                temp_EGN_noncoh_Accessories['COILPSymbolRate']=CUTLightPath.SymbolRate
                temp_EGN_noncoh_Accessories['COILPLaunchPower_dBm']=CUTLightPath.LaunchPower
                
                
#                print(iLink)
#                print(network.LinkDict[iLink].TupleDict)
                tempTupleSet=set()
                for LPIDtemp in iLink.TupleDict:
                    iLightPath=self.LightPathDict[LPIDtemp]
#                    temp_EGN_noncoh_Accessories['LinkTupleList'][LPID]=network.LinkDict[iLink].TupleDict[LPID]
                    LPTuple=(
                            iLightPath.Wavelength,
                            iLightPath.LightPathBandwidth,
                            iLightPath.SymbolRate,
                            iLightPath.roll_off_factor,
                            0
                            )
                    
                    tempTupleSet.add(LPTuple)
                    
                    if iLightPath.ModulationType=='cube4_16':
                        temp_EGN_noncoh_Accessories['ModPhiDict'][LPTuple]=-1
                        temp_EGN_noncoh_Accessories['ModPsiDict'][LPTuple]=4
                    elif iLightPath.ModulationType=='QAM4_256':
                        temp_EGN_noncoh_Accessories['ModPhiDict'][LPTuple]=-.68
                        temp_EGN_noncoh_Accessories['ModPsiDict'][LPTuple]=2.08
                    else:
                        raise Exception('Code not complete for other types of modulations!')
                    
                    temp_EGN_noncoh_Accessories['LPLaunchPowerDict'][LPTuple]=iLightPath.LaunchPower
                    
                temp_EGN_noncoh_Accessories['LinkTupleList'].append(tempTupleSet)
            
                EGN_noncoh_Accessories.append(temp_EGN_noncoh_Accessories)
            
            return EGN_noncoh_Accessories
                    
        else:
            raise Exception('Unknown return type from Export function')
#%%
'''
I think one of the optimum approaches to completing the network-level SSFM is 
that a tuple must be created consisting (mainSignal, lenSignal, UpsamplingRatio, SymbolRate).
This tuple is updated (partially or generally) at each node to make the signal ready for
the next link transmission.

*UpsamplingRatio* is the ratio of the fsampling to the CUTSymbolRate.
'''
def SNR_SSFM(network,LPID,LenStep,del_margin_percent=0.2):
    
    '''
    This block is designed to produce results in a single-link, multi-channel
    scenario.
    '''
    if type(LPID)==list:
        SingleLPID=LPID[0]
    else:
        SingleLPID=LPID
    
    CUTLightPath   = network.LightPathDict[SingleLPID]
    CUTNodeList    = CUTLightPath.NodeList
    CUTLinkList    = CUTLightPath.LinkList
    CUTSrcNode     = network.NodeDict[CUTNodeList[0]]
    CUTTransceiver = CUTSrcNode.TransceiverDict[CUTLightPath.SrcTransceiverID] #Src Transceiver of CUT
    CUTSymbolRate  = CUTTransceiver.SymbolRate
    CUTn_sym       = CUTTransceiver.TxModule.n_sym
    
    print('='*40)
    print('LP {} (Channel Under Test)\n------------\nNodeList = {}'.format(
            SingleLPID,
            CUTNodeList,
            ))
    
    Total_Signal=0
    
    set_of_all_LP_bandwidths=[(LP.Wavelength,LP.LightPathBandwidth) for LP in network.LightPathDict.values()]
    
    F_MIN             = min(set_of_all_LP_bandwidths)[0]-min(set_of_all_LP_bandwidths)[1]/2
    TOTALWDMBANDWIDTH = max(set_of_all_LP_bandwidths)[0]+max(set_of_all_LP_bandwidths)[1]/2-F_MIN
    
    UpSamplingRatio = ceil(TOTALWDMBANDWIDTH/2/CUTSymbolRate)*2
    fsampling       = CUTSymbolRate*UpSamplingRatio
    
    lenCUTSignal   = int(UpSamplingRatio*CUTn_sym)
    
    freq_D_hat=arange(lenCUTSignal)/lenCUTSignal-F_MIN/fsampling
    
    freq_D=fsampling*(freq_D_hat-floor(freq_D_hat))+F_MIN#-floor(F_MIN/fsampling)*fsampling
    
#    freq_D=arange(lenCUTSignal)/lenCUTSignal*fsampling
    
    time_D=arange(lenCUTSignal)/fsampling
    
    for CurrentLinkID,PreviousLinkID in zip(CUTLinkList,[None]+CUTLinkList):
        
        print('\n===========================================')
        print('Link {} of CUT'.format(CurrentLinkID))
        print('Total Num Syms = {}'.format(CUTTransceiver.TxModule.n_sym))
        print('WDM = {}\nsps = {}'.format(TOTALWDMBANDWIDTH/1e9,UpSamplingRatio))
        
        CurrentLink        = network.LinkDict[CurrentLinkID]
        CurrentLinkLPIDSet = set(CurrentLink.TupleDict.keys())
        
#        if CUTLinkList.index(CurrentLinkID)==0:
        if PreviousLinkID==None:
            PreviousLinkLPIDSet = set()
        else:
            PreviousLink        = network.LinkDict[PreviousLinkID]
            PreviousLinkLPIDSet = set(PreviousLink.TupleDict.keys())
            DropLPIDSet         = PreviousLinkLPIDSet-CurrentLinkLPIDSet
            
            DropFilter_f        = ones(lenCUTSignal)
            for tempLPID in DropLPIDSet:
                tempLP          = network.LightPathDict[tempLPID]
                f_min_at_tempLP = tempLP.Wavelength-tempLP.LightPathBandwidth/2
                f_max_at_tempLP = tempLP.Wavelength+tempLP.LightPathBandwidth/2
                DropFilter_f[(f_min_at_tempLP<=freq_D)*(freq_D<=f_max_at_tempLP)]=0
            
            Total_Signal=ifft(fft(Total_Signal)*DropFilter_f)
            
#            '''Plot Area'''
#            plt.figure()
#            plt.plot(fft(Total_Signal1[0])/max(fft(Total_Signal1[0])))
#            plt.plot(freq_D_at_PreviousLink/max(freq_D_at_PreviousLink))
#            plt.plot(DropFilter_f)
#            plt.title('Drop filter at link {}'.format(CurrentLinkID))
#            plt.figure()
#            plt.plot(fft(Total_Signal[0])/max(fft(Total_Signal[0])))
#            plt.plot(freq_D_at_PreviousLink/max(freq_D_at_PreviousLink))
#            plt.plot(DropFilter_f)
#            plt.title('Drop filter at link {} after drop'.format(CurrentLinkID))
#            '''========='''
            
        for tempLPID in CurrentLinkLPIDSet-PreviousLinkLPIDSet:
            
#            print(tempLPID)
            
            tempLP=network.LightPathDict[tempLPID]
            
            print('-'*40)
            print('Constructing signal from LPID {} with NodeList = {}'.format(
                    tempLPID,
                    tempLP.NodeList
                    ))
            
            '''Not important. Only checking if the lightpath traverses the link'''
            Link_index_in_tempLP=tempLP.LinkList.index(CurrentLinkID)
            '''================================================================'''
            SrcNode_of_tempLP=network.NodeDict[tempLP.NodeList[0]]
            
            LPSignal=SrcNode_of_tempLP.Total_addedSignal(
                    CUTSymbolRate,
                    {tempLP.SrcTransceiverID},
                    UpSamplingRatio,
                    lenCUTSignal
                    )
            
#            plt.figure()
#            plt.plot(LPSignal[0]*exp(-2j*pi*tempLP.Wavelength*time_D))
#            plt.figure()
#            LPSignal,useless1=SrcNode_of_tempLP.Total_addedSignal(
#                    CUTSymbolRate,
#                    {tempLP.SrcTransceiverID},
#                    UpSamplingRatio,
#                    lenCUTSignal
#                    )
            
            LPSignal_f=fft(LPSignal)
            print('Signal constructed at LP {} srcNode.'.format(tempLPID))
            for jLinkID in tempLP.LinkList:
                if tempLP.LinkList.index(jLinkID)<Link_index_in_tempLP:
                    jLink=network.LinkDict[jLinkID]
                    LPSignal_f=LPSignal_f*jLink.NetGain*exp(2j*pi**2*jLink.AccDispersion*freq_D**2)
                    print('LP {} signal passed through link {}'.format(tempLPID,jLinkID))
                else:
                    break
            LPSignal=ifft(LPSignal_f)
            
#            temp_out.append(fft(LPSignal))
#            plt.figure()
#            plt.plot(fft(LPSignal[0]))
#            plt.title('LP {} , lambda = {}'.format(tempLPID,tempLP.Wavelength/70e9))
#            plt.show()
            
            Total_Signal=Total_Signal+LPSignal
                
#            except ValueError:
#                pass
        
        print('-'*40)
        print('SSFM solver is operating at link {}...'.format(CurrentLinkID))
        
        Total_Signal,useless1,useless2,useless3=CurrentLink.NLSE_Solver_Link(
                Total_Signal,
                freq_D,
                CUTSymbolRate,
                UpSamplingRatio,
                LenStep
                )
        
#        x1=Total_Signal1-Total_Signal
#        
#        plt.figure()
#        plt.plot(real(x1[0]))
#        plt.figure()
#        plt.plot(imag(x1[0]))
#        plt.figure()
#        plt.plot(real(x1[1]))
#        plt.figure()
#        plt.plot(imag(x1[1]))
#        plt.figure()
#        
#        Total_Signal=Total_Signal1
        
        print('SSFM operation is done at link {}.'.format(CurrentLinkID))
        
#        plt.figure()
#        plt.plot(fft(Total_Signal[0])/max(fft(Total_Signal[0]))*max(freq_D_at_CurrentLink))
#        plt.plot(freq_D_at_CurrentLink)
#        
#        plt.title('Total Link Signal and freq_D at Link {}\nLink Tuple = {}'.format(
#                CurrentLinkID,
#                array(list(CurrentLink.TupleDict.values()))[:,0]/CUTLightPath.LightPathBandwidth
#                ))
#        
#        plt.show()
    
#    Total_Signal_copy_1=Total_Signal.copy()
    
    '''Time to detect!'''
    SNR_Empirical=[]
    
    if not type(LPID)==list:
        LPID=[LPID]
    
    for single_lpid in LPID:
        
#        Total_Signal_copy=Total_Signal_copy_1.copy()
        
        print('Detecting at LPID = {}'.format(single_lpid))
        
        CUTLightPath      = network.LightPathDict[single_lpid]
        CUTNodeList       = CUTLightPath.NodeList
        CUTLinkList       = CUTLightPath.LinkList
        CUTSrcNode        = network.NodeDict[CUTNodeList[0]]
        CUTSrcTransceiver = CUTSrcNode.TransceiverDict[CUTLightPath.SrcTransceiverID] #Src Transceiver of CUT
        CUTSymbolRate     = CUTSrcTransceiver.SymbolRate
        CUTn_sym          = CUTSrcTransceiver.TxModule.n_sym
#        CUTPower          = CUTLightPath.LaunchPower
#        CUTDownSamplingRatio = int(fsampling_at_CurrentLink/2/CUTSymbolRate)
        CUTDownSamplingRatio = int(UpSamplingRatio/2)
        
        Receiver_of_CUTLightPath = network.NodeDict[CUTNodeList[-1]].TransceiverDict[CUTLightPath.DestTransceiverID].RxModule
        
        ''' Transmitted symbols '''
        CUTLightPathTxSymbols    = CUTSrcTransceiver.TxModule.ModulationSymbols
        
        CUTLightPathTotalAccDispersion = sum([
                network.LinkDict[LPLinkID].AccDispersion for LPLinkID in CUTLinkList
                ])
        
#        print('Fs = {}\n'.format(fsampling))
        
#        time_D=arange(len(Total_Signal[0]))/fsampling
    #    plt.figure()
    #    plt.plot(time_D_after_Resampling)
        
        ''' Removing dispersion '''
        Total_Signal_at_EDC_output=Receiver_of_CUTLightPath.EDC(
                Total_Signal,
                CUTLightPathTotalAccDispersion,
                freq_D
                )
        
#        plt.figure()
#        plt.plot(Total_Signal_at_EDC_output[0]-Total_Signal[0])
#        plt.figure()
        
        ''' Bringing CUT to baseband '''
        Total_Signal_to_baseband=Receiver_of_CUTLightPath.to_baseband(
                Total_Signal_at_EDC_output,
                time_D
    #            time_D_after_Resampling
                )
        
        ''' Picking CUT by passing through analog low-pass filter '''
        Total_Signal_to_baseband_filtered=Receiver_of_CUTLightPath.AnalogLowPassFilter(
                Total_Signal_to_baseband,
                fsampling
                )
        
#        plt.figure()
#        plt.plot(Total_Signal_to_baseband_filtered[0])
#        plt.figure()
        
    #    Total_Signal_to_baseband_filtered,filter_order,w,h,h_linear,filt_taps=Receiver_of_DestNode.AnalogLowPassFilter(
    #            Total_Signal_to_baseband,
    #            fsampling_at_CurrentLink
    #            )
        
        ''' Downsampling the analog low-pass filter output for detecting symbols '''
        Total_Signal_to_baseband_filtered_DownSampled=Receiver_of_CUTLightPath.MF_DownSample(
                Total_Signal_to_baseband_filtered,
                CUTDownSamplingRatio
                )
        
#        plt.figure()
#        plt.plot(Total_Signal_to_baseband_filtered_DownSampled[0])
#        plt.figure()
#        Total_Signal_to_baseband_filtered_DownSampled=Receiver_of_CUTLightPath.Downsample(
#                Total_Signal_to_baseband_filtered,
#                fsampling
#                )
    #    Total_Signal_at_EDC_output=Receiver_of_CUTLightPath.EDC(
    #            Total_Signal_to_baseband_filtered_DownSampled,
    #            fsampling_at_CurrentLink,
    #            CUTLightPathTotalAccDispersion
    #            )
        
        ''' Detecting noisy symbols '''
        RxModulationSymbols=Receiver_of_CUTLightPath.SymbolDetector(
                Total_Signal_to_baseband_filtered_DownSampled,
                CUTn_sym
                )
        
        ''' Detected noisy symbols phase correction by means of the transmitted symbols '''
        RxModulationSymbols_rot=Receiver_of_CUTLightPath.PhaseRotationCompensator(
                RxModulationSymbols,
                CUTLightPathTxSymbols
                )
        
        del_margin=int(del_margin_percent*len(CUTLightPathTxSymbols[0])/2)
        
        CUTLightPathTxSymbols=array([
                CUTLightPathTxSymbols[0][del_margin:-del_margin],
                CUTLightPathTxSymbols[1][del_margin:-del_margin]
                ])
        
        RxModulationSymbols_rot=array([
                RxModulationSymbols_rot[0][del_margin:-del_margin],
                RxModulationSymbols_rot[1][del_margin:-del_margin]
                ])
        
        TxSyms_x,TxSyms_y=CUTLightPathTxSymbols
        RxSyms_x_rot,RxSyms_y_rot=RxModulationSymbols_rot
        
#        plt.figure()
#        plt.plot(real(TxSyms_x),imag(TxSyms_y),'b.')
#        plt.plot(real(RxSyms_x_rot),imag(RxSyms_y_rot),'r.')
#        plt.figure()
        
        ran_gain_restored_x=real(sum(TxSyms_x*conj(RxSyms_x_rot))/sum(abs(TxSyms_x)**2))
        ran_gain_restored_y=real(sum(TxSyms_y*conj(RxSyms_y_rot))/sum(abs(TxSyms_y)**2))
        
        ran_gain_restored=(ran_gain_restored_y+ran_gain_restored_x)/2
        
#        power_restored=mean(abs(RxSyms_x_rot)**2)+mean(abs(RxSyms_y_rot)**2)
        
        RxSyms=array([RxSyms_x_rot,RxSyms_y_rot])
        
        TxSyms=array([TxSyms_x,TxSyms_y])
#        print(power_restored)
#        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',ran_gain_restored)
#        error=TxSyms*ran_gain_restored-RxSyms
        
        error=TxSyms-RxSyms/ran_gain_restored
#        power_restored=power_restored/4/CUTDownSamplingRatio**2
#        power_restored/=ran_gain_restored
        
        var_error=sum(var(error,1))
        
        power_restored=sum(var(RxSyms,1))/ran_gain_restored**2-var_error
        
#        power_restored-=var_error
#        power_restored=10**(CUTPower*.1-3)
        
#        print('power..........',power_restored)
#        print('\n')
##        print('error.........',error)
##        print('\n')
#        print('ran_gain.........',ran_gain_restored)
#        print('\n')
#        
#        print('var_err..............',var_error)
#        
#        print(error)
        
        ''' sum(rrc_filt_res**2) = 2*UpRatio '''
        
        Single_SNR_Empirical = power_restored/var_error
        
        SNR_Empirical.append(Single_SNR_Empirical)
    
    if len(LPID)==1:
        return 10*log10(SNR_Empirical[0])#,TxSyms,RxSyms,ran_gain_restored
    else:
        return 10*log10(SNR_Empirical)
#%%
if __name__=='__main__':
    plt.close('all')
    n_sym=10000
#    PowerSet_dBm=arange(0,1,1)
#    PowerSet_dBm=arange(-5,11,1)
    roll_off_factor=0.02
    ##### Span Parameters #####
    alpha=0.22/4.343*1e-3*1
#    beta2=-21e-27*1
    beta2=-21.3e-27*1
    gamma=1.3e-3*0
#    gamma=1.3e-3*1
    Lspan=100e3
    NF=5
    ############################
    ##### Link Parameters ######
    Nspan=1
    ############################
    ###### Transceiver parameters ############
    SymbolRate=32e9
    ChannelBandwidth=32.64e9
    ############################
    COI=0
    
    power_dBm=-5
    
    ''' Defining network '''
    network=Network()
    
    ''' Adding nodes to the network '''
    for i in range(22):
        network.addNode(i)
        
    ''' Adding transceivers per nodes '''
    for iNode in network.NodeDict.values():
        for j in range(13):
            iNode.addNewTransceiver(
                    j,
#                    (j+1)*ChannelBandwidth,
                    SymbolRate,
                    ChannelBandwidth
                    )
            
    ''' Setting transceivers Tx params '''
    for iNode in network.NodeDict.values():
        for j in range(13):
            iNode.TransceiverDict[j].setTxParams(
                    n_sym,
                    roll_off_factor,
                    'cube4_16'
                    )
    
    ''' Adding links to the network '''
    for iNode in network.NodeDict:
        for jNode in network.NodeDict:
            if not iNode==jNode:
                network.addLink(iNode,jNode)
#                network.addLink(network.NodeDict[i],network.NodeDict[j])
    
    ''' Adding spans per link '''
    for iLink in network.LinkDict.values():
        for j in range(Nspan):
            iLink.addSpan(j,Lspan,alpha,beta2,gamma,NF)
            
#    network.addLightPath(0,[5,6,7,12]   ,5*32.32e9,power_dBm)
#    network.addLightPath(1,[1,5,6,8]    ,7*32.32e9,power_dBm)
#    network.addLightPath(2,[2,5,6]      ,2*32.32e9,power_dBm)
#    network.addLightPath(3,[4,3,5,6,10] ,8*32.32e9,power_dBm)
#    network.addLightPath(4,[8,6,7,11]   ,3*32.32e9,power_dBm)
#    network.addLightPath(5,[10,6,7,12]  ,7*32.32e9,power_dBm)
#    network.addLightPath(6,[3,5,6,7,11] ,11*32.32e9,power_dBm)
#    
#    u=network.Export(0,'EGN')
#    ux=network.Export(0,'EGN_noncoh')
    
    for lpid in range(1):
#        network.addLightPath(lpid,[0,1],(lpid+1)*ChannelBandwidth,power_dBm)
        network.addLightPath(lpid,list(range(21)),(lpid+1)*ChannelBandwidth,power_dBm)
#        network.addLightPath(lpid,[0,1],(lpid+1)*ChannelBandwidth,power_dBm)
    
    #%%
    _NMC=10e5
    u=network.Export(0,'EGN')
    ux=network.Export(0,'EGN_noncoh')
#    snr_egn=SNR_EGN(u,NLI_terms=None,_NMC=_NMC,printlog=True)
#    snr_egn_noncoh=SNR_EGN_noncoh(ux,NLI_terms=None,_NMC=_NMC,printlog=False)
    snr_ssfm=SNR_SSFM(network,0,LenStep=100)
#    snr_ssfm,tx,rx,pg=SNR_SSFM(network,0,LenStep=100)
#    snr_ssfm=SNR_SSFM(network,[0,1,2,3,4,5,6,7,8,9],LenStep=100)
    
    #%%
#    plt.close('all')
    
    plt.plot(snr_ssfm,'d')
#    plt.plot(snr_egn,'*')
#    plt.plot(snr_egn_noncoh,'.')
    
#    plt.figure()
    
#    rx1=rx/pg
    
#    plt.plot(real(rx1)[0],imag(rx1)[0],'b.')
#    plt.plot(real(tx)[0],imag(tx)[0],'r.')
#    
#    plt.figure()
#    plt.plot(real(rx1)[0],'.')
#    plt.plot(real(tx)[0],'.')
    
#    plt.figure()
    
#    plt.plot(snr_ssfm)
    
#    SNR_SSFM_dB = []
#    SNR_EGN_dB  = []
#    SNR_EGN_link_dB  = []
#    SNR_CFM_dB  = []
#    
#    for power_dBm in PowerSet_dBm:
#        print('\n*********************************'*2)
#        print('Power (dBm) = {}'.format(power_dBm))
#        print('*********************************\n'*2)
#        network.delallLightPath()
#        ''' Adding lightpaths '''
#        network.addLightPath(0,[5,6,7,12]   ,5*32.32e9,power_dBm)
#        network.addLightPath(1,[1,5,6,8]    ,7*32.32e9,power_dBm)
#        network.addLightPath(2,[2,5,6]      ,2*32.32e9,power_dBm)
#        network.addLightPath(3,[4,3,5,6,10] ,8*32.32e9,power_dBm)
#        network.addLightPath(4,[8,6,7,11]   ,3*32.32e9,power_dBm)
#        network.addLightPath(5,[10,6,7,12]  ,7*32.32e9,power_dBm)
#        network.addLightPath(6,[3,5,6,7,11] ,11*32.32e9,power_dBm)
#        
#        EGN_Accessories=network.Export(COI,'EGN')
#        CFM_Accessories=network.Export(COI,'CFM')
#        
#        snr_egn_temp,snr_egn_temp_link,__=SNR_EGN(EGN_Accessories)
#        snr_cfm_temp,_=SNR_CFM(CFM_Accessories,'CFM4')
##        
##        
#        SNR_EGN_dB.append(snr_egn_temp)
#        SNR_CFM_dB.append(snr_cfm_temp)
#        SNR_EGN_link_dB.append(snr_egn_temp_link)
###    CFM_Accessories=network.Export(COI,'CFM')
###    snr_cfm_temp,_=SNR_CFM(CFM_Accessories,'CFM1')
#    plt.figure()
#    plt.plot(SNR_EGN_dB,label='EGN')
#    plt.plot(SNR_CFM_dB,label='CFM')
#    plt.plot(SNR_EGN_link_dB,label='link')
#    plt.legend()
#        network.Export(COI,'CFM')
#    SNR_CFM(x)
#    print(
#            I(
#                    x['f_comb'],
#                    x['R_comb'],
#                    x['f_CUT'],
#                    32.64e9,
#                    x['CUTindex_comb'],
#                    x['alpha'],
#                    x['beta2'],
#                    x['Length'],
#                    beta3spanvec=None,
#                    f_c=None,
#                    model_kind='CFM4'
#                    )
#            )
#        snr_temp,a,b=SNR_EGN(EGN_Accessories)
        
#    SNR_SSFM_dB.append(SNR_SSFM(network,COI,100))
#    plt.figure()
#    plt.plot(SNR_SSFM_dB)
#        SNR_EGN_dB.append(snr_temp)
    
#    plt.figure()
#    plt.plot(PowerSet_dBm,SNR_EGN_dB,label='EGN')
#    plt.plot(PowerSet_dBm,SNR_SSFM_dB,'o',label='SSFM')
#    plt.xlabel('Power (dBm)')
#    plt.ylabel('SNR (dB)')
#    plt.legend(fontsize=13)
#    plt.savefig('SNR_EGN_SSFM.png',dpi=1000)
#    plt.savefig('SNR_EGN_SSFM.eps')
#    plt.show()