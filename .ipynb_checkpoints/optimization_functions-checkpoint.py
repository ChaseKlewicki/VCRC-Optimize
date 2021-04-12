import numpy as np
import CoolProp.CoolProp as CP
from cycle_functions import *
from scipy.optimize import minimize, Bounds, NonlinearConstraint, LinearConstraint
import warnings
import pandas as pd
import traceback

def make_cycle(Vars, Inputs, Param, refrigerant = 'R410a'):

    # ----------------------------------------------#
    # ==------ Vars  -------==#
    P_c    = Vars[0] # Pa
    P_e    = Vars[1] # Pa
    T_SH   = Vars[2] # delta-T K
    RPM    = Vars[3]
    RPM_cond = Vars[4]
    RPM_evap = Vars[5]
    # ----------------------------------------------#
    #==------ Inputs ------==#
    
    T_amb  = Inputs[0] # K
    T_pod  = Inputs[1] # K
    Q_load = Inputs[2] # W
    
    #----------------------------------------------#
    #==------ Param -------==#
    
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

    if T_SH < 0:
        T_SH =1e-4
    
    # pressure drop accross evaporator (Pa)
    delta_P_e = 0
    
    # pressure drop accross condenser (Pa)
    delta_P_c = 0
    
    P[0] = P_e - delta_P_e # Pressure drop accross evap determined empirically
    
    
    # Init state
    P_crit = CP.PropsSI('Pcrit', refrigerant)
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
    eta_is = 5.15
    
    if 1 / eta_is < 0 or 1 / eta_is > 1:
        warnings.warn('Infeasible isentropic Efficiency: ' + str(eta_is))
   
    h[1] = h[0] + (CP.PropsSI('H', 'P', P_c, 'S', s[0], refrigerant) - h[0]) / eta_is
    s[1] = CP.PropsSI('S', 'P', P[1], 'H', h[1], refrigerant)

    STATE = [P[1], h[1]]
    
    #   calculate condenser
    [P[1:5], T[1:5], h[1:5], s[1:5], abscissa[1:5], W_fan_c] = Condenser_Proc( STATE, 
                                                             'h', m_dot_s, T_amb, delta_P_c, RPM_cond, refrigerant)

    #   calculate expansion
    m_dot_v = capillary_tube_func(P[4], T[4], h[4], refrigerant)
    
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
    Q_def  =  (Q_load - Q_L) / Q_load   #Pod energy deficit

    Deficit = np.array([m_def, h_def, Q_def])

    #Other Outputs
    m_dot = [m_dot_s, m_dot_v]
    
    # Combined efficiency (Regression determined empirically)
    eta_comb = P_crit / P_e *  0.00416802 + 0.01495443
    
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
        [_, _, _, _, _, _, _, _, _, _, _, COSP, _] = make_cycle(Vars, Inputs, Param)
        Obj = -COSP
        return Obj
                        
    #
    #
    # Make Nonlinear Constraint for T_SH

    def nonlcon1(Vars):
        c = (T_pod - CP.PropsSI('T', 'P', Vars[1], 'Q', 0, refrigerant)) - Vars[2] 
        return c
    
    def nonlcon2(Vars):
        [_, _, _, _, _, _, _, _, _, _, _, _, Deficit] = make_cycle(Vars, Inputs, Param)
        return np.abs(Deficit[1])
    
    def nonlcon3(Vars):
        [_, _, _, _, _, _, _, _, _, _, _, _, Deficit] = make_cycle(Vars, Inputs, Param)
        return np.linalg.norm([Deficit[0], Deficit[2]])

    nonLinear1 = NonlinearConstraint(nonlcon1, 0, np.inf)
    nonLinear2 = NonlinearConstraint(nonlcon2, 0, 0.001)
    nonLinear3 = NonlinearConstraint(nonlcon3, 0, 0.03)
    
    a = np.identity(6)[0:3,:]
    linear1 = LinearConstraint(A = a,
                               lb = [CP.PropsSI('P', 'T', T_amb, 'Q', 1, refrigerant), 
                                    300e3, 
                                    0,], # Lower Bounds
                               ub = [CP.PropsSI('PCRIT', refrigerant), 
                                    CP.PropsSI('P', 'T', T_pod, 'Q', 0, refrigerant), 
                                    30,], # Upper Bounds
                                keep_feasible=True)
    
    a = np.identity(6)[3:6,:]
    linear2 = LinearConstraint(A = a,
                               lb = [1500,
                                    750,
                                    500], # Lower Bounds
                               ub = [2000,
                                    2900,
                                    2900], # Upper Bounds
                                keep_feasible=False)
    
    # Solve the problem.
    try:
        res = minimize(objective, Vars, constraints = [nonLinear1, nonLinear2, nonLinear3, linear1, linear2], 
                           method = 'trust-constr', options = {'maxiter': 10000})
    except ValueError as e:
        print(traceback.format_exc())
        print('initial Point: ' + str(Vars))
        res = {'success': False, 'x': Vars}
        
    print(res)
        
    # ---
    if res['success']:
        Vars = res['x']
        [_, _, _, _, _, _, _, _, _, _, _, COSP, _] = make_cycle(Vars, Inputs, Param)
    else:
        print('failed')
        Vars = res['x']
        [_, _, _, _, _, _, _, _, _, _, _, COSP, _] = make_cycle(Vars, Inputs, Param)

    return [Vars, COSP]