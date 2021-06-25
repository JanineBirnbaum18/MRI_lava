# Literature theoretical and empirical models of rheological parameters with varying
# volume fractions of solid particles and gas bubbles suspended in liquid

import numpy as np
from scipy.special import erf

# Relative visocisty (consistency) in particle suspensions

def Einstein(phi, B=2.5):
    # B = 2.5 for rigid, monodisperse spheres
    return 1 + B*phi

def Einstein_Roscoe(phi, B=2.5, b=1, phi_m = 1):
    # B is Einstein coefficient
    # b = 1 for monodisperse, b = 1.35 for polydisperse (empirical)
    return (1 - b*phi/phi_m)**(-B)

def Mooney(phi, B=2.5):
    # B is Einstein coefficient
    return np.exp(B*phi/(1-phi))

def Krieger_Dougherty(phi, B=2.5, phi_m=0.55):
    return (1 - phi/phi_m)**(-B*phi_m)

def Costa(phi, B=2.5, phi_crit=0.5, xi=0.0005, delta=7, gamma=5):
    F = (1 - xi)*erf(np.sqrt(np.pi)/(2*(1-xi))*phi/phi_crit*((1+(phi/phi_crit)**gamma)))
    return (1 + (phi/phi_crit)**delta)/(1-F)**(B*phi_crit)

# Relative visocisty (consistency) in gas suspensions

def Taylor(phi, B=1):
    return 1 + B*phi

def Llewellin_Manga(phi, Ca, K=6/5, m=2):
    #KCa can be approximated by Cx
    eta_r0 = (1-phi)**(-1)
    eta_rinf = (1-phi)**(5/3)
    return eta_rinf + (eta_r0 - eta_rinf)/(1 + (K*Ca)**m)

def Princen_Kiss_K(phi, Ca):
    return 32*(phi-0.73)*Ca**(-1/2)

# Relative visocisty (consistency) in three-phase suspensions
# X =V_solid/V_total, Y = V_gas/V_total

# From Phan-Thien & Pham (1997)
def PTP1(X, Y, solid_crit, B=1, cmax=100):
    case_1 = ((1 - X/(solid_crit*(1-Y)))**(-5/2))*((1 - Y)**(-B))
    case_1[case_1 > cmax] = cmax
    return case_1

def PTP2(X, Y, solid_crit, cmax=100):
    case_2 = (1 - X/solid_crit - Y)**(-(5*X/solid_crit + 2*Y)/(2*(X/solid_crit + Y)))
    case_2[case_2 > cmax] = cmax
    return case_2

def PTP3(X, Y, solid_crit, B=1, cmax=100):
    case_3 = ((1 - Y/(1 - X))**(-B))*((1 - X/solid_crit)**(-5/2))
    case_3[case_3 > cmax] = cmax
    return case_3

# Flow index in solid-liquid suspensions

def Castruccio_n(phi, phi_crit=0.27, phi_m=0.6, C=1.3):
    n = np.ones_like(phi)
    n[phi>phi_crit] += C*((phi_crit-phi[phi>phi_crit])/phi_m)
    return n

def Mueller_n(phi,phi_m=0.633,rp=1):
    return 1-0.2*rp*(phi/phi_m)**4

# Flow index in gas-liquid suspensions

def Truby_n_gas(phi, C=0.334):
    return 1 - C*phi

# Yield Stress in solid-liquid suspensions

def Hoover(phi, phi_crit=0.25, phi_m=0.525, A=5.3, p=1):
    tauy = A*(((phi/phi_crit)-1)/(1-phi/phi_m))**(1/p)
    tauy[phi>phi_m] = np.inf
    return tauy

def Mueller_tauy(phi, tau_star=0.153, phi_m=0.633):
    tauy = tau_star*(1 - phi/phi_m)**(-2) - 1
    tauy[phi>phi_m] = np.inf
    return tauy

def Castruccio_tauy(phi, phi_crit=0.27, D=5*10**6):
    tauy = np.zeros_like(phi)
    tauy[phi>phi_crit] += D*(phi[phi>phi_crit] - phi_crit)**8
    return tauy

# Yield stress in gas-liquid suspensions (actually emulsions)

def Princen_Kiss_tauy(phi, sigma=0.08, r=0.001):
    Y = -0.080 - 0.114*np.log(1-phi)
    return sigma*phi**(1/3)*Y/r

# This study

def Birnbaum_Lev_K(phi_s, phi_g, phi_m=0.55, B_solid=2.5, B_gas=2):
    K = ((1 - phi_s/(phi_m*(1-phi_g)))**(-B_solid))*((1 - phi_g)**(-B_gas))
    if type(K) != np.ndarray:
        if (phi_s/(1-phi_g)>phi_m)>phi_m:
            K = np.inf
    else:
        K[phi_s/(1-phi_g)>phi_m] = np.inf
    return K

def Birnbaum_Lev_K_real(phi_s, phi_g, phi_m=0.55, B_solid=2.5, B_gas=2):
    K = ((1 - phi_s/(phi_m*(1-phi_g)))**(-B_solid))*((1 - phi_g)**(-B_gas))
    K[phi_s/(1-phi_g)>phi_m] = 1e12
    return K

def Birnbaum_Lev_tauy(phi_s, phi_g, C1=70, C2=7, phi_c = 0.3):
    return 10**(C1*(phi_s/(1-phi_g) - phi_c)) + 10**(C2*(phi_s/(1-phi_g) + phi_g - phi_c)) #- 10**(-C1*phi_c) - 10**(-C2*phi_c) 

def Birnbaum_Lev_n(phi_s, phi_g, Ca, C3=0.87, C4=0.5, phi_c = 0.3):
    n = 1 + (phi_c - phi_s - phi_g)*(C3 - C4*Ca)
    n[(phi_s + phi_g < phi_c)] = 1
    return n

    
    