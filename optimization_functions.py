import numpy as np
import CoolProp.CoolProp as CP
from cycle_functions import *
from scipy.optimize import minimize, Bounds, NonlinearConstraint, LinearConstraint
import warnings
import pandas as pd

def make_cycle(Vars, Inputs, Param, refrigerant = 'R410a'):

    # ----------------------------------------------#
    # ==------ Vars  -------==#
    P_c    = Vars[0] # Pa
    P_e    = Vars[1] # Pa
    T_SH   = Vars[2] # delta-T K
    # ----------------------------------------------#
    #==------ Inputs ------==#
    
    T_amb  = Inputs[0] # K
    T_pod  = Inputs[1] # K
    Q_load = Inputs[2] # W
    
    #----------------------------------------------#
    #==------ Param -------==#
    RPM    = Param[0]
    RPM_cond = Param[1]
    RPM_evap = Param[2]
    
    #----------------------------------------------#
    #==-- Init. Outputs  --==#
    P = np.zeros(9) # Pa
    T = np.zeros(9) # K
    h = np.zeros(9) # j/kg
    s = np.zeros(9) # j/kg/k
    abscissa = np.zeros(9)
    # var "abscissa" is the nondimensional 
    # Heat exchanger position 
    # for each of these stations
    # domain = [0,1]U[1,2]
    # [0,1] <-- in condensor
    # [1,2] <-- in evaporator
    
    #=========================================================================#
    # Calculate
    #=========================================================================#

    # pressure drop accross evaporator (Pa)
    delta_P_e = 0
    
    # pressure drop accross condenser (Pa)
    delta_P_c = 0
    
    P[0] = P_e - delta_P_e # Pressure drop accross evap determined empirically
    
#     if T_SH < 0:
#         print('Super heat input negative. Changed value to zero.')
#         T_SH = 0
    
    
    # Init state
    T_sat_e = CP.PropsSI('T', 'P', P[0], 'Q', 1, refrigerant) # K
    h_g     = CP.PropsSI('H', 'P', P[0], 'Q', 1, refrigerant) # J/kg
    T[0] = T_sat_e + T_SH
    h[0] = CP.PropsSI('H', 'P', P[0], 'T', T[0], refrigerant)
    abscissa[0] = 0
    s[0] = CP.PropsSI('S', 'P', P[0], 'H', h[0], refrigerant)
    
    STATE   = [P[0], h[0]]
    
    #   calculate compressor
    m_dot_s = compr_func(STATE, RPM, P_c / P[0])
    P[1] = P_c
    
    # Isentropic Ratio
    eta_is = 2.9
   
    h[1] = h[0] + (CP.PropsSI('H', 'P', P_c, 'S', s[0], refrigerant) - h[0]) / eta_is
    s[1] = CP.PropsSI('S', 'P', P[1], 'H', h[1], refrigerant)

    STATE = [P[1], h[1]]
    
    #   calculate condenser
    [P[1:5], T[1:5], h[1:5], s[1:5], abscissa[1:5], W_fan_c] = Condenser_Proc( STATE, 
                                                             'h', m_dot_s, T_amb, delta_P_c, RPM_cond, refrigerant)

    #   calculate expansion
    m_dot_v = capillary_tube_func(P[4], h[4], T[4], refrigerant)
    
    P[5] = P_e
    # Isenthalpic expansion
    h[5] =  h[4]
    
    STATE = [P[5], h[5]]
    

    #   calculate evap
    [P[5:9], T[5:9], h[5:9], s[5:9], abscissa[5:9], W_fan_e] = Evap_Proc(STATE, m_dot_v, T_pod, delta_P_e, RPM_evap, refrigerant)

    abscissa[5:9] = abscissa[5:9] + abscissa[4]

    # Energy and Mass Deficits
    Q_L = m_dot_v * (h[8] - h[5])
    Q_H = m_dot_s * (h[1] - h[4])

    m_def  =  (m_dot_s - m_dot_v) / m_dot_s  #Mass Deficit
    h_def  =  10 * (h[0]  - h[8]) / h[0]   # evap deficit
    Q_def  =  (Q_L  - Q_load) / Q_load   #Pod energy deficit

    Deficit = np.array([m_def, h_def, Q_def])

    #Other Outputs
    m_dot = [m_dot_s, m_dot_v]
    
    # Combined efficiency (Regression determined empirically)
    eta_comb = 1 / (P_c / P_e * 59.84697757 - 150.16285207)
    
    #     P_ratio = []
    #     for index, row in modelData.iterrows():
    #         P_ratio.append(row['P (Pa)'][1] / row['P (Pa)'][0])
    #     plt.plot(P_ratio, experimentalData['Compressor Work (W)'] / modelData['Compressor Work (W)'], 'o')
    #     np.polyfit(P_ratio, experimentalData['Compressor Work (W)'] / modelData['Compressor Work (W)'], 1)
    #     plt.show()
    
    # Compute compressor work based on isentropic, adiabatic compressor
    W_comp     = m_dot_s * (h[1] - h[0]) / eta_comb

    # Compute Coefficient of system performance
    COSP = Q_L / (W_comp + W_fan_c + W_fan_e)

    return [P, T, h, s, abscissa, m_dot, Q_L, Q_H, W_comp, W_fan_c, W_fan_e, COSP, Deficit]


def adjust_cycle_fmin(Vars, Inputs, Param, refrigerant = 'R410a'):

    assert(np.size(Vars) == 3)

    T_amb  = Inputs[0]
    T_pod  = Inputs[1]

    #
    #
    # Make Objective Function

    def objective(Vars):
        [_, _, _, _, _, _, _, _, _, _, _, _, Obj] = make_cycle(Vars, Inputs, Param)
        
        Obj = 1000 * np.linalg.norm(Obj)
        
        return Obj
                        
    #
    #
    # Make Nonlinear Constraint for T_SH

    def nonlcon(Vars):
        c = (T_pod - CP.PropsSI('T', 'P', Vars[1], 'Q', 0, refrigerant)) - Vars[2] 
        return c

    nonLinear = NonlinearConstraint(nonlcon, 0, np.inf)
    
    linear = LinearConstraint(A =np.concatenate((np.identity(3), np.array([[1, -3, 0]])), axis=0),
                              lb = [CP.PropsSI('P', 'T', T_amb, 'Q', 1, refrigerant), 200e3, 0.1, -np.inf], # Lower Bounds
                              ub = [5000e3, CP.PropsSI('P', 'T', T_pod, 'Q', 0, refrigerant), 30, 0] # Upper Bounds
                             )

    #
    # Solve the problem.
    try:
        res = minimize(objective, Vars, constraints = [nonLinear, linear], 
                       method = 'trust-constr', options = {'maxiter': 500})
    except ValueError as e:
        print(e)
        print('initial Point: ' + str(Vars))
        res = {'success': False}
    
    # ---
    if res['success']:
        Vars = res.x
        [_, _, _, _, _, _, _, _, _, _, _, _, Deficit] = make_cycle(Vars, Inputs, Param)
    else:
        Deficit = [1, 1, 1]

    return [Vars, Deficit]


def solve_cycle_shotgun(Inputs, Param, refrigerant = 'R410a'):
    
    T_amb  = Inputs[0] # K
    T_pod  = Inputs[1] # K
    
    SPREAD = 4;

    # evaporator bounds
    lb = [200e3, CP.PropsSI('P', 'T', T_amb, 'Q', 1, refrigerant)] # lower bound for evap and cond Pressures
    ub = [CP.PropsSI('P', 'T', T_pod, 'Q', 0, refrigerant), 3] # upper bound for evap and compression ratio bound for cond
    
    # Initial guess for superheat
    T_SH  = 0.5
    
    # Intialize Vars
    Vars = np.empty((0,3))
    
    # Create list of Initial points in feasible region
    P_e   = lb[0] + (ub[0] - lb[0]) * np.linspace( 0.1, 0.9, SPREAD)
    for P in P_e:
        P_c = P + (P * ub[1] - lb[1]) * np.linspace( 0.1, 0.9, SPREAD)
        Vars = np.concatenate([Vars, np.array(np.meshgrid(P_c, P, T_SH)).T.reshape(-1,3)])


    #Initialize Vars and Deficits
    normDeficit = np.zeros(len(Vars))
    Deficit     = np.zeros((len(Vars), 3))

    # Try different initial points
    for ind, Var in enumerate(Vars):
        #Step Vars Forward
        [Vars[ind], Deficit[ind]] = adjust_cycle_fmin( Var, Inputs, Param)
        normDeficit[ind] = np.linalg.norm(Deficit[ind])
        
    
    # find solution with lowest error
    Vars = Vars[normDeficit == np.nanmin(normDeficit)][0]
    
    # Check if error is lower than 3% 
    converged = 1
    if normDeficit[normDeficit == np.nanmin(normDeficit)] > 0.05:
        converged = 0
        warnings.warn('Warning: |Deficit| = ' + 
                      str(normDeficit[normDeficit == min(normDeficit)]))

    #Calc
    [P, T, h, s, abscissa, m_dot, Q_L, Q_H, W_comp, W_fan_c, W_fan_e, COSP, Deficit] = make_cycle(Vars, 
                                                                                             Inputs,
                                                                                             Param)
    Props = [P, T, h, s, abscissa]
        
    return [Props, m_dot, Q_L, Q_H, W_comp, W_fan_c, W_fan_e, COSP, Deficit, converged]