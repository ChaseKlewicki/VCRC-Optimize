import numpy as np
import CoolProp.CoolProp as CP
from cycle_functions import *
from scipy.optimize import minimize, Bounds, NonlinearConstraint, LinearConstraint
import warnings
import pandas as pd

def make_cycle(Vars, Inputs, Param, refrigerant = 'R410a'):

    # ----------------------------------------------#
    # ==------ Vars  -------==#
    RPM    = Vars[0]
    RPM_cond = Vars[1]
    RPM_evap = Vars[2]
    
    # ----------------------------------------------#
    #==------ Inputs ------==#
    T_amb  = Inputs[0] # K
    T_pod  = Inputs[1] # K
    Q_load = Inputs[2] # W
    
    # ----------------------------------------------#
    #==------ Params ------==#
    P_c    = Param[0] # Pa
    P_e    = Param[1] # Pa
    T_SH   = Param[2] # delta-T K
    
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
    
    
    # Init state
    T_sat_e = CP.PropsSI('T', 'P', P[0], 'Q', 1, refrigerant) # K
    h_g     = CP.PropsSI('H', 'P', P[0], 'Q', 1, refrigerant) # J/kg
    T[0] = T_sat_e + T_SH
    h[0] = CP.PropsSI('H', 'P', P[0], 'T', T[0], refrigerant)
    abscissa[0] = 0
    s[0] = CP.PropsSI('S', 'P', P[0], 'H', h[0], refrigerant)
    
    STATE   = [P[0], h[0]]
    
    #   calculate compressor
    m_dot_s = compr_func(STATE, RPM, P_c)
    P[1] = P_c
    
    # Isentropic Ratio
    eta_is = 5.15 * RPM / 2000
    
    if 1 / eta_is < 0 or 1 / eta_is > 1:
        warnings.warn('Infeasible isentropic Efficiency: ' + str(eta_is))
   
    h[1] = h[0] + (CP.PropsSI('H', 'P', P_c, 'S', s[0], refrigerant) - h[0]) / eta_is
    s[1] = CP.PropsSI('S', 'P', P[1], 'H', h[1], refrigerant)

    STATE = [P[1], h[1]]
    
    #   calculate condenser
    [P[1:5], T[1:5], h[1:5], s[1:5], abscissa[1:5], W_fan_c] = Condenser_Proc( STATE, 
                                                             'h', m_dot_s, T_amb, delta_P_c, RPM_cond, refrigerant)

    #   calculate expansion
    m_dot_v = capillary_tube_func(P[4], T[4], refrigerant)
    
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
    W = m_dot_s * (h[1] - h[0])

    m_def  =  (m_dot_s - m_dot_v) / m_dot_s  #Mass Deficit
    h_def  =  (h[0] - h[8]) / h[0]# h deficit
    Q_def  =  (Q_L  - Q_load) / Q_load   #Pod energy deficit

    Deficit = np.array([m_def, h_def, Q_def])

    #Other Outputs
    m_dot = [m_dot_s, m_dot_v]
    
    # Combined efficiency (Regression determined empirically)
    eta_comb = (P_c / P_e * 0.0343053 - 0.048315)
    
#     if eta_comb < 0 or eta_comb > 1:
#         warnings.warn('Infeasible Combined Efficiency: ' + str(eta_comb))
    
    # Compute compressor work based on isentropic, adiabatic compressor
    W_comp = m_dot_s * (h[1] - h[0]) / eta_comb

    # Compute Coefficient of system performance
    COSP = Q_L / (W_comp + W_fan_c + W_fan_e)

    return [P, T, h, s, abscissa, m_dot, Q_L, Q_H, W_comp, W_fan_c, W_fan_e, COSP, Deficit]


def adjust_cycle_fmin(Vars, Inputs, Param, refrigerant = 'R410a'):

    assert(np.size(Vars) == 6)

    T_amb  = Inputs[0]
    T_pod  = Inputs[1]

    #
    #
    # Make Objective Function

    def objective(Vars):
        [_, _, _, _, _, _, _, _, _, _, _, _, Obj] = make_cycle(Vars, Inputs, Param)
        
        Obj = 1000 * np.linalg.norm(Obj)
        print(Obj)
        return Obj
                        
    #
    #
    # Make Nonlinear Constraint for T_SH

    def nonlcon(Vars):
        c = (T_pod - CP.PropsSI('T', 'P', Vars[1], 'Q', 0, refrigerant)) - Vars[2] 
        return c

    nonLinear = NonlinearConstraint(nonlcon, 0, np.inf)
    
    linear = LinearConstraint(A = np.identity(3),
                              lb = [0,
                                    0,
                                    0], # Lower Bounds
                              ub = [2000,
                                    10000,
                                    10000], # Upper Bounds
                              keep_feasible=True)

    # Solve the problem.
    try:
        res = minimize(objective, Vars, constraints = [nonLinear, linear], 
                       method = 'trust-constr', options = {'maxiter': 1000})
    except ValueError as e:
        print(e)
        print('initial Point: ' + str(Vars))
        res = {'success': False}
    
    print(res)
    
    # ---
    if res['success']:
        Vars = res.x
        [_, _, _, _, _, _, _, _, _, _, _, _, Deficit] = make_cycle(Vars, Inputs, Param)
    else:
        Deficit = [1, 1, 1]

    return [Vars, Deficit]