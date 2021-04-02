import numpy as np
import warnings
import CoolProp.CoolProp as CP
from scipy.optimize import fsolve


def compr_func( inlet_state, RPM, P_c, refrigerant = 'R410a'):
    

    P_e   = inlet_state[0] # Pa
    h_e_o = inlet_state[1] # j/kg
        
        
    # Critical Pressure
    P_crit = CP.PropsSI('Pcrit', refrigerant)

    # Volumetric efficiency
    eta_v = P_crit / P_c * 0.11486726 + 0.33963123

    Disp = 5.25E-6    # [m^3 per rev] #volume displacement
    
    if eta_v < 0:
        raise ValueError('Compression ratio too high:')

    h_g   = CP.PropsSI('H', 'P', P_e, 'Q', 1, refrigerant)
    if h_e_o < h_g:
        warnings.warn('Flooded Compressor, vapor quality < 1')
    
    rho = CP.PropsSI('D', 'P', P_e, 'H' ,h_e_o, refrigerant)

    m_dot = RPM / 60 * Disp * eta_v * rho
    
    return m_dot


def Gnielinski_Nu(Re, Pr, d = None, epsilon = None):
    # Correlation for Nusselt number in pipes
    
    # Check that non dimensional parameters are in valid range
    if Pr < 0.5 or Pr > 2000 or Re < 2300 or Re > 5e6:
        raise ValueError('Gnielinski Not Valid for Re or Pr Re: ' + str(Re) + ' Pr: ' + str(Pr))
    
    if d == None or epsilon == None:
        # Smooth pipe correlation
        f_2 = (1.58 * np.log(Re) - 3.28)**-2
    else:  
        # Compute friction factor from Colebrook Eqn
        f_2 = 1
        f_1 = 0
        while np.abs(f_1 - f_2) > 1e-5:
            f_1 = f_2
            f_2 = 1 / (2 * np.log10(2.51 / (Re * np.sqrt(f_1)) + (epsilon / d) / 3.72))**2
    
    # Compute nusselt number
    Nu = (f_2 / 2 * (Re - 1000) * Pr) / (1 + 12.7 * np.sqrt(f_2 / 2) * (Pr**(2 / 3) -1))
    
    return Nu


def dittus_boelter(Re, Pr, HX):
    
    if HX == 'COND':
        Nu = 0.023 * Re**0.8 * Pr*0.3
    if HX == 'EVAP':
        Nu = 0.023 * Re**0.8 * Pr*0.4
    
    return Nu


def generate_HTCOEFF(P, m_dot_i, subsys, T_o, RPM, x_in, refrigerant = 'R410a'):

    if subsys == 'COND':
        
        # Guess for head pressure loss across air side of HX
        guess = 0
        
        # Initialize dp
        dP = 1
        
        # instability control
        num_of_iters = 0
        
        while np.abs(guess - dP) > 0.1:
            
            num_of_iters = num_of_iters + 1
            
            if num_of_iters > 1000:
                raise ValueError('Volume flow rate Calc Divergent ' + subsys + " " + str(RPM))
            
            # Use new dP as next guess
            guess = dP
            
            # Compute fan work and volumetric flow rate based on fan rpm
            [V_dot_o, W_fan] = HT_900(RPM, dP)
            
            # Shroud efficiency
            eta_shroud = 1
            
            V_dot_o = V_dot_o * eta_shroud

            #-------------------------------------------------------------------------#
            # Outside (air side)
            #-------------------------------------------------------------------------#

            # Geometric Characteristics

            # Fin density (fins/m) [measured 19 fins per inch]
            Nf = 19 / 0.0254

            # Outside diameter of tubing (m) [measured 1/4"]
            do = 1 / 4 * 0.0254

            # Inside diameter of tubing (m) [wall thickness measured at 0.03"]
            di = do - 2 * 0.03 * .0254

            # Transverse spacing between tubes (m) [measured 1.048" - do]
            xt = 1.048 * 0.0254 - do

            # Longitudinal spacing between tubes (m) [(measured 1.066" - do) / 2]
            xl = (1.066 * 0.0254 - do) / 2

            # Fin thickness (m) [measured 0.004"]
            delta = 0.004 * 0.0254

            # Overall Length (m) [measured 15 + 15/16 ] (parially blocked by compressor!)
            L1 = (8) * 0.0254

            # Overall depth (m) [measured 1.5]
            L2 = (1.5) * 0.0254

            # Overall height (m) [measured 12.5"] (parially blocked by headers)
            L3 = (8) * 0.0254

            # Number of Rows 
            Nr = 3

            # Number of tubes. Two tube slots are empty
            Nt = 44 - 2

            # Primary surface area (tubes and header plates)
            A_p = np.pi * do * (L1 - delta * Nf * L1) * Nt + 2 * (L3 * L2 - np.pi * do**2 / 4 * Nt)

            # Secondary surface area (fins)
            A_f = 2 * (L2 * L3 - (np.pi * do**2 / 4) * Nt) * Nf * L1 + 2 * L3 * delta * Nf * L1 

            # Heat transfer area airside (m^2)
            A_o = A_f + A_p 

            # Volume occupied by the heat exchanger (heat exchanger total volume) (m^3)
            V_o = L1 * L2 * L3

            # Minimum free-flow area (Fundamentals of Heat Exchanger Design-Shah pg 573) 

            # 2a''
            a_prime = (xt - do) - (xt - do) * delta * Nf 

            # 2b''
            b_prime = 2 * (((xt / 2) ** 2 + xl ** 2) ** 0.5 - do - (xt - do) * delta * Nf)

            # c''
            if a_prime < b_prime:
                c_prime = a_prime
            else:
                c_prime = b_prime

            # Minimum free-flow area (m^2)
            A_o_o = ((L3 / xt - 1) * c_prime + (xt - do) - (xt - do) * delta * Nf) * L1

            # Frontal area (m^2)
            A_fr_o = L1 * L3

            # Ratio of free flow area to frontal area
            sigma_o  = A_o_o / A_fr_o

            # surface area density 
            alpha_o = A_o / V_o

            # Hydralic diameter (m)
            D_h_o = 4 * sigma_o / alpha_o

            # Mean velocity accross HX (m/s)
            V_o = V_dot_o / A_fr_o

            # Maximum velocity inside the HX (m/s)
            V_max_o = V_o / sigma_o 

            #-------------------------------------------------------------------------#
            # Air Constants
            #-------------------------------------------------------------------------#

            k_o = CP.PropsSI('L', 'P', 101325, 'T', T_o, 'air') #[W/m-K]   
            mu_o = CP.PropsSI('V', 'P', 101325, 'T', T_o, 'air') #[Pa-s]   
            rho_o = CP.PropsSI('D', 'P', 101325, 'T', T_o, 'air') #[kg/m3] 
            c_p_o = CP.PropsSI('C', 'P', 101325, 'T', T_o, 'air') #[J/kg-K]
            Pr_o = CP.PropsSI('Prandtl', 'P', 101325, 'T', T_o, 'air') #[J/kg-K]

            #-------------------------------------------------------------------------#
            # Derived Relations
            #-------------------------------------------------------------------------#

            # Mass flow rate of air (kg/s)
            m_dot_o = V_dot_o * rho_o

            # refrigerant mass velocity (kg/(m^2 s))
            G_o = V_o * rho_o 

            # Compute Reynold's number
            Re_o = G_o * D_h_o / mu_o

            # Compute j using equation 7.141 Nr >= 2 (Fundamentals of Heat Exchanger Design-Shah pg 551)

            # collar diameter (m)
            dc = do + 2 * delta

            # Hydralic Diameter described by correlation
            D_h = 4 * A_o_o * L2 / A_o

            # Collar Reynolds number
            Re_dc = rho_o * V_max_o * dc / mu_o

            # fin pitch (m/fin)
            pf = 1 / Nf

            # constants from equation
            C3 = -0.361 - 0.042 * Nr / np.log(Re_dc) + 0.158 * np.log(Nr * (pf / dc) **0.41)

            C4 = -1.224 - 0.076 * (xl / D_h) ** 1.42 / np.log(Re_dc)

            C5 = -0.083 + 0.058 * Nr / np.log(Re_dc)

            C6 = -5.735 + 1.21 * np.log(Re_dc / Nr)

            # Compute outside heat transfer coefficeinet using coburn j factor (more accurate)
            j = 0.086 * Re_dc ** C3 * Nr ** C4 * (pf / dc) ** C5 * (pf / D_h) ** C6 * (pf / xt) ** -0.93

            # exponents from correlation
            F1 = -0.764 + 0.739 * (xt / xl) + 0.177 * (pf / dc) - 0.00758 / Nr

            F2 = -15.689 + 64.021 / np.log(Re_dc)

            F3 = 1.696 - 15.695 / np.log(Re_dc)

            # Compute friction factor of HX for air side
            f = 0.0267 * Re_dc**F1 * (xt / xl)**F2 * (pf / dc)**F3

            # Compute head pressure assuming negligible density change
            dP = f * A_o / A_o_o * (rho_o * V_max_o)**2 / 2 / rho_o
        
        # h = JGCp/Pr^2/3
        h_o = j * V_o * c_p_o * rho_o / Pr_o ** (2/3)
        
        # radius of tube including collar thickness
        r_e = dc / 2

        # Single fin efficiency 
        # (Fundamentals of Heat Exchanger Design-Shah pg 606 eqn 9.14)
        
        # Pipe Wall thermal conductivity [copper] (W/m K)
        k_pipe = 385 
        
        m = (2 * h_o / k_pipe / delta) ** 0.5
        
        # geometric parameter for schidt fin efficieny approx.
        xm = xt / 2

        # equivelent fin radius
        R_eq = r_e * (1.27 * xm / r_e * ((np.sqrt((xt / 2)**2 + xl**2) / 2) / xm - 0.3)**0.5)
        
        # Phi parameter for staggered arrangement
        phi  = (R_eq / r_e - 1) * (1 + 0.35 * np.log(R_eq / r_e))
        
        # Determine single fin efficiency
        eta_f = np.tanh(m * r_e * phi) / (m * r_e * phi)
        
        #Overall Fin efficiency
        fin_eff = (1 - (1 - eta_f) * A_f / A_o)
        
        #-------------------------------------------------------------------------#
        # Inside (refrigerant side)
        #-------------------------------------------------------------------------#
        
        # Geometric Characteristics
        
        # Heat transfer area refrig. side (m^2)
        A_i = np.pi * di * Nt * L1 # [m2]

        # Conduction resistance of pipe wall (K/W)
        R_tw = np.log(do / di) / (2 * np.pi * Nt * L1 * k_pipe) 
        
        #-------------------------------------------------------------------------#
        # Refrigerant Constants (R410a) 
        #-------------------------------------------------------------------------#

        k_f  = CP.PropsSI('L', 'P', P, 'Q', 0, refrigerant) # [W/m-K] 
        k_g  = CP.PropsSI('L', 'P', P, 'Q', 1, refrigerant) # [W/m-K] 
        mu_f = CP.PropsSI('V', 'P', P, 'Q', 0, refrigerant) # [Pa-s] 
        mu_g = CP.PropsSI('V', 'P', P, 'Q', 1, refrigerant) # [Pa-s]
        c_p_f = CP.PropsSI('C', 'P', P, 'Q', 0, refrigerant) #[J/kg-K]  
        c_p_g = CP.PropsSI('C', 'P', P, 'Q', 1, refrigerant) #[J/kg-K]
        
        #-------------------------------------------------------------------------#
        # Derived Relations
        #-------------------------------------------------------------------------#

        #HT-coefficient, gaseous, contribution from refrigerant side
        Re_g  =  4 * m_dot_i / (np.pi * di * mu_g)
        Pr_g  =  c_p_g * mu_g / k_g
        
        # Turbulent
        if Re_g > 3000:
            Nu_g  =  dittus_boelter(Re_g, Pr_g, subsys)
        # Transition
        elif Re_g < 3000 and Re_g > 2300:
            Nu_g = (dittus_boelter(Re_g, Pr_g, subsys) * (Re_g- 2300) + 3.66 * (3000 - Re_g)) / 700
        # Laminar
        else:
            Nu_g = 3.66  
            
        h_i_g =  Nu_g * k_g / di


        #HT-coefficient, liquid, contribution from refrigerant side
        Re_f  =  4 * m_dot_i / (np.pi * di * mu_f)
        Pr_f  =  c_p_f * mu_f / k_f
        
        # Turbulent
        if Re_f > 3000:
            Nu_f  = dittus_boelter(Re_f, Pr_f, subsys)
        # Transition
        elif Re_f < 3000 and Re_f > 2300:
            Nu_f = (dittus_boelter(Re_f, Pr_f, subsys) * (Re_f - 2300) + 3.66 * (3000 - Re_f)) / 700
        # Laminar
        else:
            Nu_f = 3.66
            
        h_i_f =  Nu_f * k_f / di
        
        # Two phase HT-coefficient
        h_i_TP  = condensation(h_i_f, P, refrigerant)

        # Overall HT-coefficient
        UA_g = (1 / (h_i_g * A_i) + R_tw + 1 / (fin_eff * h_o * A_o))**-1
        UA_TP = (1 / (h_i_TP * A_i) + R_tw + 1 / (fin_eff * h_o * A_o))**-1
        UA_f = (1 / (h_i_f * A_i) + R_tw + 1 / (fin_eff * h_o * A_o))**-1
        
        # Determine HT effectiveness
        epsilon = 0.4
        
        # Apply effectiveness
        UA_g = epsilon * UA_g
        UA_TP = epsilon * UA_TP
        UA_f = epsilon * UA_f
        
        #Local overall heat transfer coefficient
        U_g = UA_g / A_i
        U_TP = UA_TP / A_i
        U_f = UA_f / A_i
        
    elif subsys == 'EVAP':
        
        # Guess for head pressure loss across air side of HX
        guess = 0
        
        # Initialize dp
        dP = 1
        
        # instability control
        num_of_iters = 0
        
        while np.abs(guess - dP) > 0.1:
            
            num_of_iters = num_of_iters + 1
            
            if num_of_iters > 1000:
                raise ValueError('Volume flow rate Calc Divergent ' + subsys + " " + str(RPM))
                
            # Use new dP as next guess
            guess = dP
            
            # Compute fan work and volumetric flow rate based on fan rpm
            [V_dot_o, W_fan] = blower(RPM, guess)
            
            # Shroud efficiency
            eta_shroud = 1
            
            V_dot_o = V_dot_o * eta_shroud
            
            #-------------------------------------------------------------------------#
            # Outside (air side)
            #-------------------------------------------------------------------------#

            # Geometric Characteristics

            # Fin density (fins/m) [measured 20 fins per inch]
            Nf = 19 / 0.0254

            # Outside diameter of tubing (m) [measured .31"]
            do = 3 / 8 * 0.0254

            # Inside diameter of tubing (m) [wall thickness estimated at 0.03"]
            di = do - 2 * 0.03 * 0.0254

            # Transverse spacing between tubes (m) [measured 0.86"]
            xt = 0.86 * 0.0254

            # Longitudinal spacing between tubes (m) [(measured 0.994") / 2]
            xl = (0.994 * 0.0254) / 2

            # Fin thickness (m) [measured 0.004"]
            delta = 0.004 * 0.0254

            # Overall Length (m) 
            L1 = (12.5) * 0.0254

            # Overall depth (m) [measured 1.5"]
            L2 = (1.5) * 0.0254

            # Overall height (m) 
            L3 = 8.5 * 0.0254

            # Number of Rows 
            Nr = 3

            # Number of tubes
            Nt = 30

            # Primary surface area (tubes and header plates)
            A_p = np.pi * do * (L1 - delta * Nf * L1) * Nt + 2 * (L3 * L2 - np.pi * do**2 / 4 * Nt)

            # Secondary surface area (fins)
            A_f = 2 * (L2 * L3 - (np.pi * do**2 /  4) * Nt) * Nf * L1 + 2 * L3 * delta * Nf * L1 

            A_o = A_f + A_p #[m2] #Heat transfer area airside

            # Volume occupied by the heat exchanger (heat exchanger total volume) (m^3)
            V_o = L1 * L2 * L3

            # Minimum free-flow area (Fundamentals of Heat Exchanger Design-Shah pg 573) 

            # 2a''
            a_prime = (xt - do) - (xt - do) * delta * Nf 

            # 2b''
            b_prime = 2 * (((xt / 2) ** 2 + xl ** 2) ** 0.5 - do - (xt - do) * delta * Nf)

            # c''
            if a_prime < b_prime:
                c_prime = a_prime
            else:
                c_prime = b_prime

            # Minimum free-flow area (m^2)
            A_o_o = ((L3 / xt - 1) * c_prime + (xt - do) - (xt - do) * delta * Nf) * L1

            # Frontal area (m^2)
            A_fr_o = L1 * L3

            # Ratio of free flow area to frontal area
            sigma_o  = A_o_o / A_fr_o

            # surface area density 
            alpha_o = A_o / V_o

            # Hydralic diameter (m)
            D_h_o = 4 * sigma_o / alpha_o

            # Mean velocity accross HX (m/s)
            V_o = V_dot_o / A_fr_o

            # Maximum velocity inside the HX (m/s)
            V_max_o = V_o / sigma_o

            #-------------------------------------------------------------------------#
            # Air Constants
            #-------------------------------------------------------------------------#

            k_o = CP.PropsSI('L', 'P', 101325, 'T', T_o, 'air') # [W/m-K]   
            mu_o = CP.PropsSI('V', 'P', 101325, 'T', T_o, 'air') # [Pa-s]   
            rho_o = CP.PropsSI('D', 'P', 101325, 'T', T_o, 'air') # [kg/m3] 
            c_p_o = CP.PropsSI('C', 'P', 101325, 'T', T_o, 'air') # [J/kg-K]
            Pr_o = CP.PropsSI('Prandtl', 'P', 101325, 'T', T_o, 'air') # [J/kg-K]

            #-------------------------------------------------------------------------#
            # Derived Relations
            #-------------------------------------------------------------------------#

            # Mass flow rate of air (kg/s)
            m_dot_o = V_dot_o * rho_o

            # refrigerant mass velocity (kg / (m^2 s))
            G_o =  V_o * rho_o 

            # Compute Reynold's number
            Re_o = G_o * D_h_o / mu_o

            # Compute j using equation 7.141 Nr >= 2 (Fundamentals of Heat Exchanger Design-Shah pg 551)

            # collar diameter (m)
            dc = do + 2 * delta

           # Hydralic Diameter described by correlation
            D_h = 4 * A_o_o * L2 / A_o

            # Collar Reynolds number
            Re_dc = rho_o * V_max_o * dc / mu_o

            # fin pitch (m / fin)
            pf = 1 / Nf

            # exponenets from correlation
            C3 = -0.361 - 0.042 * Nr / np.log(Re_dc) + 0.158 * np.log(Nr * (pf / dc)**0.41)

            C4 = -1.224 - (0.076 * (xl / D_h)**1.42) / np.log(Re_dc)

            C5 = -0.083 + 0.058 * Nr / np.log(Re_dc)

            C6 = -5.735 + 1.21 * np.log(Re_dc / Nr)

            # Compute outside heat transfer coefficeinet using colburn j factor (more accurate)
            j = 0.086 * Re_dc**C3 * Nr**C4 * (pf / dc)**C5 * (pf / D_h)**C6 * (pf / xt)**-0.93

            # exponents from correlation
            F1 = -0.764 + 0.739 * (xt /xl) + 0.177 * (pf / dc) - 0.00758 / Nr

            F2 = -15.689 + 64.021 / np.log(Re_dc)

            F3 = 1.696 - 15.695 / np.log(Re_dc)

            # Compute friction factor of HX for air side
            f = 0.0267 * Re_dc**F1 * (xt / xl)**F2 * (pf / dc)**F3

            # Compute head pressure assuming negligible density change
            dP = f * A_o / A_o_o * (rho_o * V_max_o)**2 / 2 / rho_o
        
        # h = JGCp/Pr^2/3
        h_o = j * V_o * rho_o * c_p_o / Pr_o**(2/3)

        # radius of tube including collar thickness
        r_e = dc / 2

        # Single fin efficiency 
        # (Fundamentals of Heat Exchanger Design-Shah pg 606 eqn 9.14)
        
        # Pipe Wall thermal conductivity [copper] (W/m K)
        k_pipe = 385
        
        m = (2 * h_o / k_pipe / delta) ** 0.5
        
        # geometric parameter for schmidt fin efficieny approx.
        xm = xt / 2

        # equivelent fin radius
        R_eq = r_e * (1.27 * xm / r_e * ((np.sqrt((xt / 2)**2 + xl**2) / 2) / xm - 0.3)**0.5)
        
        # Phi parameter for staggered arrangement
        phi  = (R_eq / r_e - 1) * (1 + 0.35 * np.log(R_eq / r_e))
        
        # Determine single fin efficiency
        eta_f = np.tanh(m * r_e * phi) / (m * r_e * phi)
        
        #Overall Fin efficiency
        fin_eff = 1 - (1 - eta_f) * A_f / A_o
        
        #-------------------------------------------------------------------------#
        # Inside (refrigerant side)
        #-------------------------------------------------------------------------#
        
        # Geometric Characteristics
        
        # Heat transfer area refrig. side (m^2)
        A_i = np.pi * di * Nt * L1
        
        # conduction heat transfer through wall
        R_tw = np.log(do / di) / (2 * np.pi * k_pipe * Nt * L1) # (K / W)
        
        #-------------------------------------------------------------------------#
        # Refrigerant Constants (R410a) 
        #-------------------------------------------------------------------------#

        k_f  = CP.PropsSI('L', 'P', P, 'Q', 0, refrigerant) # (W/m-K) 
        k_g  = CP.PropsSI('L', 'P', P, 'Q', 1, refrigerant) # (W/m-K) 
        mu_f = CP.PropsSI('V', 'P', P, 'Q', 0, refrigerant) # (Pa-s) 
        mu_g = CP.PropsSI('V', 'P', P, 'Q', 1, refrigerant) # (Pa-s)
        c_p_f = CP.PropsSI('C', 'P', P, 'Q', 0, refrigerant) # (J/kg-K )
        c_p_g = CP.PropsSI('C', 'P', P, 'Q', 1, refrigerant) # (J/kg-K)
        h_fg = (CP.PropsSI('H', 'P', P, 'Q', 1, refrigerant) - 
                CP.PropsSI('H', 'P', P, 'Q', 0, refrigerant)) # (J/kg)

        # HT-coefficient, gaseous, contribution from refrigerant side (W/m^2 K)
        Re_g  =  4 * m_dot_i / (np.pi * di * mu_g)
        Pr_g  =  c_p_g * mu_g / k_g
        
        # Turbulent
        if Re_g > 3000:
            Nu_g  =  dittus_boelter(Re_g, Pr_g, subsys)
        # Transition
        elif Re_g < 3000 and Re_g > 2300:
            Nu_g = (dittus_boelter(Re_g, Pr_g, subsys) * (Re_g- 2300) + 3.66 * (3000 - Re_g)) / 700
        # Laminar
        else:
            Nu_g = 3.66  
            
        h_i_g =  Nu_g * k_g / di


        #HT-coefficient, liquid, contribution from refrigerant side
        Re_f  =  4 * m_dot_i / (np.pi * di * mu_f)
        Pr_f  =  c_p_f * mu_f / k_f
        
        # Turbulent
        if Re_f > 3000:
            Nu_f  = 0.21 * Re_f**0.62 * Pr_f**0.4
        # Transition
        elif Re_f < 3000 and Re_f > 2300:
            Nu_f = (0.21 * Re_f**0.62 * Pr_f**0.4 * (Re_f - 2300) + 3.66 * (3000 - Re_f)) / 700
        # Laminar
        else:
            Nu_f = 3.66
            
        h_i_f =  Nu_f * k_f / di
        
        # Two phase HT-coefficient (W/m^2 K)
        h_i_TP  = boiling(h_i_f, P, m_dot_i * h_fg * (1 - x_in), refrigerant, 
                          m_dot_i / (np.pi * (di / 2)**2) , do, x_in)

        
        # Overall HT-coefficient
        UA_g = (1 / (h_i_g * A_i)  + R_tw + 1 / (fin_eff * h_o * A_o))**-1
        UA_TP = (1 / (h_i_TP * A_i) + R_tw + 1 / (fin_eff * h_o * A_o))**-1
        UA_f = (1 / (h_i_f * A_i) + R_tw + 1 / (fin_eff * h_o * A_o))**-1
        
        # Determine HT effectiveness
        epsilon = 0.51
        
        # Apply effectiveness
        UA_g = epsilon * UA_g
        UA_TP = epsilon * UA_TP
        UA_f = epsilon * UA_f

        #Local overall heat transfer coefficient
        U_g = UA_g / A_i
        U_TP = UA_TP / A_i
        U_f = UA_f / A_i

    else:
        raise ValueError('Subsys must be "COND" or "EVAP"')
        
        
    return [UA_g, UA_TP, UA_f, W_fan]


def Condenser_Proc(input_state, strarg, flowrate, T_amb, P_drop, RPM, refrigerant = 'R410a'):


    # Input state must be a row vector containing pressure 
    # and enthalpy in that order
    # input_state = [P, h]
    
    
    #Initialize Vars
    #----------------------
    P_in = input_state[0]
    P = P_in * np.ones(4)
    h = np.zeros(4)
    T = np.zeros(4)
    s = np.zeros(4)

    abcissa = np.zeros(4)
    dz_1 = 0
    dz_2 = 0
    dz_3 = 0

    #=========================================================================#
    # set up us the properties

    if strarg == 'h':

        h_in = input_state[1]
        T_in = CP.PropsSI('T', 'P', P_in, 'H', h_in, refrigerant)
        T_sat = CP.PropsSI('T', 'P', P_in, 'Q', 1, refrigerant)
        h_f   = CP.PropsSI('H', 'P', P_in, 'Q', 0, refrigerant)
        h_g   = CP.PropsSI('H', 'P', P_in, 'Q', 1, refrigerant)
        h_fg  = h_g - h_f

        # assign output
        #----------------
        T[0] = T_in;
        h[0] = h_in;
        #----------------

    elif strarg == 'T':

        T_in = input_state[1];
        h_in = CP.PropsSI('H', 'P', P_in, 'T', T_in, refrigerant)
        T_sat = CP.PropsSI('T', 'P', P_in, 'Q', 1, refrigerant)
        h_f   = CP.PropsSI('H', 'P', P_in, 'Q', 0, refrigerant)
        h_g   = CP.PropsSI('H', 'P', P_in, 'Q', 1, refrigerant)
        h_fg  = h_g - h_f


        # assign output
        #----------------
        T[0] = T_in;
        h[0] = h_in;
        #----------------

    else:
        raise ValueError('dont recognize input property' + strarg)



    #=========================================================================#
    # Calculate Vars
    #

    [UA_g, UA_TP, UA_f, W_fan] = generate_HTCOEFF( P_in, flowrate, 'COND', T_amb, RPM, 1)
    
    
    

    #Properties
    c_p_g = 0.5 * (CP.PropsSI('C', 'P', P_in, 'H', h_in, refrigerant) + 
                   CP.PropsSI('C', 'P', P_in, 'Q', 1, refrigerant))
    
    c_p_f = CP.PropsSI('C', 'P', P_in, 'Q', 0, refrigerant)
    
    #=========================================================================#
    #
    #  begin integration procedure, piecewise
    #
    #

    #--- Superheat-into-Saturation Process ---
    # Check that ambiet temperature is above the saturation 
    # and inlet temperature otherwise go straight to subcooled
    if (T_amb - T_in) < 0  and (T_amb - T_sat) < 0:
        
        # Check if superheated
        if h_in > h_g:
            dz_1 = c_p_g  * flowrate / UA_g * - np.log((T_sat - T_amb) / (T_in - T_amb))

            #Add exception if superheated phase takes up the
            #entire HX domain
            if (dz_1 > 1):
                raise ValueError('no exception when superheated' +
                                 ' phase takes up entire domain')


            T[1] = T_sat
            h[1] = h_g
        
        # Else Two Phase
        else:
            T[1] = T_in
            h[1] = h_in
            dz_1 = 0
            
        #--- SatVap-into-SatLiq Process ---
        dz_2 = flowrate * h_fg / (UA_TP * (T_sat - T_amb))

            #Begin exception if saturation phase takes up the 
            #remainder of the HX domain
        if (dz_1 + dz_2) > 1:

            dz_2   = 1 - dz_1

            #solve system 
            dh_2 = (UA_TP * (T_sat - T_amb)) * dz_2 / flowrate

            #-----------------
            # Produce Output
            #
            h_out = h_g - dh_2;
            #
            # assign output
            #-----------------
            T[2] = T_sat
            h[2] = h_out
            T[3] = T[2]
            h[3] = h[2]
            #-----------------

            #Otherwise go to subcool process  
        else:

            # assign output
            #-----------------
            T[2] = T_sat
            h[2] = h_f
            #-----------------      



    #--- SatLiq-into-Subcool Process ---        

    dz_3 = 1 - dz_1 - dz_2

    T_out = (T_sat - T_amb) * np.exp(-UA_f / (c_p_f * 
                                              flowrate) * dz_3) + T_amb
    h_out = h_f + c_p_f * (T_out - T_sat)


    # assign output
    #-----------------
    T[3] = T_out;
    h[3] = h_out;
    
    # Pressure drop determined empirically applied linearly
    P[1] = P[0] - P_drop * dz_1
    P[2] = P[1] - P_drop * dz_2
    P[3] = P[2] - P_drop * (1 - dz_2 + dz_1)
    #-----------------


    # assign output
    #-----------------------------------
    abcissa[1] = abcissa[0] + dz_1
    abcissa[2] = abcissa[1] + dz_2
    abcissa[3] = 1
    #-----------------------------------    

    s = CP.PropsSI('S', 'P', P, 'H', h, refrigerant)


    return [P, T, h, s, abcissa, W_fan]


def Evap_Proc(input_state, flowrate, T_pod, P_drop, RPM, refrigerant = 'R410a'):


    # Input state must be a row vector containing pressure 
    # and enthalpy in that order
    # input_state = [P, h]
    

    #
    # Initialize Vars
    #----------------------
    P_in = input_state[0]
    P = P_in * np.ones(4)
    h = np.zeros(4)
    T = np.zeros(4)
    s = np.zeros(4)

    abcissa = np.zeros(4)
    dz_1 = 0
    dz_2 = 0
    dz_3 = 0


    #=========================================================================#
    # set up us the properties
    #
    h_in  = input_state[1]

    T_sat = CP.PropsSI('T', 'P', P_in, 'Q', 0, refrigerant)
    h_f   = CP.PropsSI('H', 'P', P_in, 'Q', 0, refrigerant)
    h_g   = CP.PropsSI('H', 'P', P_in, 'Q', 1, refrigerant)
    h_fg  = h_g - h_f    


    #=========================================================================#
    # Calculate Vars


    #Properties
    c_p_g = CP.PropsSI('C', 'P', P_in, 'Q', 1, refrigerant)
    c_p_f = CP.PropsSI('C', 'P', P_in, 'Q', 0, refrigerant)

    #=========================================================================#
    #
    #  begin integration procedure, piecewise
    #
    #=

    if h_in >= h_f:  #There is no subcooled region

        dz_1 = 0;
        # assign output
        #----------------
        T[0] = T_sat;
        h[0] = h_in;
        T[1] = T_sat;
        h[1] = h_in;
        #----------------
        
        
        # vapor quality
        x_in  = (h_in - h_f) / h_fg
        
        [UA_g, UA_TP, UA_f, W_fan] = generate_HTCOEFF( P_in, flowrate, 'EVAP', T_pod, RPM, x_in)
        
    else: #calculate subcooled region
        #--- Subcooled-into-SatLiq Process ---
    
        # Vapor Quality
        x_in  = 0
    
        [UA_g, UA_TP, UA_f, W_fan] = generate_HTCOEFF( P_in, flowrate, 'EVAP', T_pod, RPM, x_in)

        T_in = T_sat + (h_in - h_f) / c_p_f

        dh_1 = h_f - h_in;
        
        dz_1 = (c_p_f * flowrate / UA_f ) * np.log( (T_pod - T_in) / (T_pod - T_sat));

        # assign output
        #----------------
        T[0] = T_in;
        h[0] = h_in;
        T[1] = T_sat;
        h[1] = h_f;
        #----------------

    #--- SatLiq-into-SatVap Process ---

    dh_2 = h_g - h[1]
    dz_2 = flowrate * dh_2 / (UA_TP * (T_pod - T_sat))
    
        #Begin exception if saturation phase takes up the 
        #remainder of the HX domain
    if (dz_2) > (1 - dz_1):
        warnings.warn('Partial Evaporation')

        dz_2 = (1 - dz_1)
        
        #Solve system for dh_2
        dh_2  = (UA_TP * (T_pod - T_sat)) * dz_2 / flowrate

        #-----------------
        # Produce Output
        #
        h_out = h_in + dh_2 
        #
        # assign output
        #-----------------
        T[2] = T_sat;
        h[2] = h_out;
        T[3] = T_sat;
        h[3] = h_out;
        #-----------------


    
    else: # Otherwise go to superheat process  
        # assign output
        #-----------------
        T_sat = CP.PropsSI('T', 'P', P_in, 'Q', 1, refrigerant)
        T[2] = T_sat
        h[2] = h_g
        #-----------------      

        #--- SatLiq-into-Subcool Process ---        

        dz_3 = 1 - dz_2 - dz_1
        T_out = (T_sat - T_pod) * np.exp(-UA_g / (c_p_g * flowrate) * dz_3 ) + T_pod
        
        h_out = CP.PropsSI('H', 'T', T_out, 'P', P_in, refrigerant)

        # assign output
        #-----------------
        T[3] = T_out
        h[3] = h_out
        #-----------------
    

    # assign output
    # Pressure drop determined empirically applied linearly
    P[1] = P[0] - P_drop * (dz_1)
    P[2] = P[1] - P_drop * (dz_2)
    P[3] = P[2] - P_drop * (1 - dz_2 + dz_1)
    #-----------------------------------
    abcissa[1] = abcissa[0] + dz_1
    abcissa[2] = abcissa[1] + dz_2
    abcissa[3] = 1
    #-----------------------------------    
    
    s = CP.PropsSI('S', 'P', P, 'H', h, refrigerant)


    return [P, T, h, s, abcissa, W_fan]


def valve_func( CA_param, P_up, P_down, x):

    # CA_par : [m2]  dimensional parameter
    # P_up   : [kPa] upstream press
    # P_down : [kPa] downstream press
    # x      : [  ]  valve opening fraction

    # At 0.80 valve opening we have the rated value

    # Density 
    rho_v = CP.PropsSI('D', 'P', P_up, 'Q', 0, 'R410a')

    # Mass flow rate
    m_dot = CA_param * ( x / 0.80 ) * np.sqrt( rho_v * (P_up - P_down) )

    return  m_dot


def capillary_tube_func(P_in, T_in, h_in, refrigerant = 'R410a'):
    # Mass flow rate correlation from 1998 ASHRAE Hand Book. chp 44
    
    # Diameter of capillary tube coil 4"
    D = 4 * 0.0254

    # 1/16" in OD copper tubing, .018" wall thickness
    d = (1/16 - 2 *.0179) * 0.0254

    # length of capillary tube. 4 loops
    L = D * np.pi  * 4

    # delta subcool
    T_sub = CP.PropsSI('T', 'P', P_in, 'Q', 0, refrigerant) - T_in

    # Isobaric specific heat capacity of r-410a refrigerant at inlet pressure
    C_pf = CP.PropsSI('C', 'P', P_in, 'Q', 0, refrigerant)

    # Dynamic viscosity of r-410a refrigerant at inlet temperature
    mu_f = CP.PropsSI('V', 'T', T_in, 'Q', 0, refrigerant)

    # Dynamic viscosity of r-410a vapor at inlet temperature
    mu_g = CP.PropsSI('V', 'T', T_in, 'Q', 1, refrigerant)

    # Density of r-410a refrigerant at inlet pressure
    rho_f = CP.PropsSI('D', 'P', P_in, 'Q', 0, refrigerant)

    # Density of r-410a vapor at inlet pressure
    rho_g = CP.PropsSI('D', 'P', P_in, 'Q', 1, refrigerant)

    # Specific volume of r-410a refrigerant at inlet temperature
    v_f = 1 / rho_f

    # Specific volume of r-410a vapor at inlet temperature
    v_g = 1 / rho_g

    # Saturated liquid surface tension of r-410a vapor at inlet pressure
    sigma = CP.PropsSI('I', 'P', P_in, 'Q', 0, refrigerant)

    # Enthalpy of refrigerant at inlet pressure
    h_f = CP.PropsSI('H', 'P', P_in, 'Q', 0, refrigerant)

    # Enthalpy of vaporization at inlet pressure
    h_fg = (CP.PropsSI('H', 'P', P_in, 'Q', 1, refrigerant) - 
             CP.PropsSI('H', 'P', P_in, 'Q', 0, refrigerant))

    # A generalized continuous empirical correlation for predicting refrigerant
    # mass flow rates through adiabatic capillary tubes

    pi_1 = L / d
    pi_2 = d**2 * h_fg / v_f**2 / mu_f**2 
    pi_4 = P_in * d**2 / v_f / mu_f**2
    pi_5 = T_sub * C_pf * d**2 / v_f**2 / mu_f**2 
    pi_6 = v_g / v_f
    pi_7 = (mu_f - mu_g) / mu_g
    
    # If Two Phase: meter flow
    if T_sub <= 1: 
        warnings.warn('Cavitation in expansion valve')
#         pi_5 = (h_in - h_f) / h_fg
#         if pi_5 < 0.03:
#             pi_5 = 0.03
            
#         pi_8 = 187.27 * pi_1**-0.635 * pi_2 **-0.189  * pi_4**0.645 * pi_5**-0.163 * pi_6**-0.213 * pi_7 **-0.483 
        m_dot = 1e-5
        
    # Else subcooled: Valid from 1 to 16 C
    else:
        pi_8 = 1.8925 * pi_1**-0.484 * pi_2 **-0.824  * pi_4**1.369 * pi_5**0.0187 * pi_6**0.773 * pi_7 **0.265 
        
        # Two tubes
        m_dot = 2 * pi_8 * d * mu_f
    
    return m_dot


def HT_900(RPM, dP):
    # Fan performance based on honeywell HT_900 fan
    
    W = 40 * (RPM / 2900)**3 # W
    
    # Head Pressure intercept on fan curve (Pa)
    intercept_P = 185 * (RPM / 2900)**2
    
    # Volume flow rate intercept on fan curve (m^3/s)
    intercept_V_dot = RPM * 185 / 2900 * 0.00047194745 
    
    if dP > intercept_P:
        V_dot = 1e-8 # very small
        W = 40 * (RPM / 2900)**3
    
    else:
        # Create fan curve for given RPM
        V_dot = intercept_V_dot * (1 - (dP / intercept_P)**2) 
    
    
    return [V_dot, W]


def blower(RPM, dP):
    # Fan performance based on ebmpabst RER 190-39/14/2TDLOU fan
    
    W = 80 * (RPM / 2900)**3 # W
    
    # Head Pressure intercept on fan curve (Pa)
    intercept_P = 330 * (RPM / 2900)**2
    
    # Volume flow rate intercept on fan curve (m^3/s)
    intercept_V_dot = RPM * 330 / 2900 * 0.00047194745 
    
    if dP > intercept_P:
        V_dot = 1e-8 # very small
        W = 80 * (RPM / 2900)**3
    
    else:
        # Create fan curve for given RPM
        V_dot = intercept_V_dot * (1 - (dP / intercept_P)**2)
    
    
    return [V_dot, W]


def fan(RPM, dP):
    # Fan performance based on ebmpabst RER 190-39/14/2TDLOU fan
    
    W = 57.1 * (RPM / 2900)**3 # W
    
    # Head Pressure intercept on fan curve (Pa)
    intercept_P = 360.818 * (RPM / 2900)**2
    
    # Volume flow rate intercept on fan curve (m^3/s)
    intercept_V_dot = RPM * 376.7 / 2900 * 0.00047194745 
    
    if dP > intercept_P:
        raise ValueError('Fan Stalled dP: ' + str(dP))
    
    # Create fan curve for given RPM
    V_dot = intercept_V_dot * (1 - (dP / intercept_P)**2) 
    
    
    return [V_dot, W]


def epsilonNTU(C_i, C_o, UA):
    # Epsilon-NTU method for heat transfer effectiveness during liquid and gas refrigerant
    # phases of heat transfer. For compact fin tube HX with 3 rows
    
    # args:
    # C_i: Inside (refrigerant side) heat capacity rate m_dot_i * C_p_i (j/s-k)
    # C_o: Outside (air side) heat capacity rate m_dot_o * C_p_o (j/s-k)
    # UA: Overall Heat transfer coef. (W / K)
    
    # Outputs:
    # Epsilon: Heat Transfer Effectiveness (unitless)
    
    # Assign C_min and C_max
    if C_i > C_o:
        C_min = C_o
        C_max = C_i
        
        # Compute number of transfer units
        NTU = UA / C_min
        
        # Compute C*
        # Refrigerant is Two-Phase
        if np.isinf(C_max): 
            # Compute C*
            C_star = 0
            
            epsilon = 1 - np.exp(-NTU)
            
        # If outside side C_min and refrigerant not TP
        else:
            # Compute C*
            C_star = C_min / C_max
            
            # Use proper epsilon-NTU formula for 3 row fin tube HX 
            K = 1 - np.exp(-NTU / 3)

            
            epsilon = 1 / C_star * (1 - np.exp(-3 * K * C_star) * 
                                    (1 + C_star * K**2 * (3 - K) + 
                                     3 * C_star**2 * K**4 / 2))
            
        
    else:
        C_min = C_i
        C_max = C_o
        
        # Compute number of transfer units
        NTU = UA / C_min
        
        # Outside fluid is Two-Phase
        if np.isinf(C_max): 
            
            # Compute C*
            C_star = 0
            
            epsilon = 1-np.exp(-NTU)
            
        # If refrig side C_min and not TP
        else:
            # Compute C*
            C_star = C_min / C_max
            
            # Use proper epsilon-NTU formula for 3 row fin tube HX 
            K = 1 - np.exp(-NTU * C_star / 3)
            
            epsilon = 1 - np.exp(-3 * K / C_star) * (1 + K**2 * (3 - K) / C_star + 3 * K**4 / (2 * C_star**2))
            

    return epsilon


def boiling(h_l, P, Q, refrigerant, G, d_o, x_in = 0):
    # Shah correlation for heat transfer during boiling on 
    # bundles of horizontal plain and enhanced tubes. 
    
    F_pb = 1
    
    # Critical Pressure
    P_crit = CP.PropsSI('PCRIT', refrigerant)
    
    # Molecular weight
    M = CP.PropsSI('M', refrigerant)
    
    # Density of saturated liquid (kg/m^3)
    rho_l = CP.PropsSI('D', 'P', P, 'Q', 0, refrigerant)
    
    i_fg = (CP.PropsSI('H', 'P', P, 'Q', 1, refrigerant) - 
            CP.PropsSI('H', 'P', P, 'Q', 0, refrigerant))
    
    # gravity (m/s^2)
    g = 9.81
    
    # Refrence Pressure
    P_r = P / P_crit
    
    # Boiling number
    Bo = Q / G / i_fg
    
    # Froude Number
    Fr = G**2 / (rho_l**2 * g * d_o)
    
    # Boiling intensity parameter
    Y = Bo * Fr**0.3
    
    if x_in == 0:
        z = np.nan
    else:
        z = ((1 - x_in) / x_in)**0.8 * P_r**0.4
    
    h_cooper = 55.1 * Q**0.67 * P_r**0.12 * (-np.log(P_r))**-0.55 * M **-0.55
    
    if Y > 0.0008:
        h_b = F_pb * h_cooper
    else:
        phi = 2.3 / (z**0.08 * Fr**0.22)
        
        phi_o = np.max([1, 443 * F_pb * Bo**0.65, 31 * F_pb * Bo**0.33])
        
        h_b = np.nanmax([F_pb * h_cooper, h_l * phi, h_l * phi_o])
        
    return h_b


def condensation(h_l, P, refrigerant):
    # Shah mean heat transfer correlation for film condensation inside pipes.
    # Only valid for full phase change!! Changing from 1-0 vapor quality
    # inputs
    # h_l: liquid heat transfer coef given by Dittus-Boelter equation (w/m^2 K)
    # refrigerant: String of refrigerant that is condensing
    # P: condensation pressure (Pa)
    
    # Critical Pressure
    P_crit = CP.PropsSI('PCRIT', refrigerant)
    
    # Refrence Pressure
    P_r = P / P_crit
    
    # Condensation Heat transfer coef
    h_c = h_l * (0.55 + 2.09 / P_r**0.38)
    
    return h_c