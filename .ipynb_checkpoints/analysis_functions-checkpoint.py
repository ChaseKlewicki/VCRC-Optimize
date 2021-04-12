import numpy as np
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import pandas as pd
import os


# Create fit function for pressure transducers using pressure cal coefficients
pressureFit = np.poly1d([125 * 6.89476,  -62.5 * 6.89476]) * 1000 

def experimental_analysis(file, P_amb, Q_element, W_refrig, V = np.nan, offset = 2):
    # A function which performs analysis of steady state values for 
    # the pressure and temperature measurements taken during the prototype experiments
    
    # file: Location of experimental data file.
    # P_amb: Ambient pressure measured by altimeter. (Pa)
    # Q_element: Input Power to heating element. measured by Power Meter (Wh)
    # W_refrig: Power consumed by blower and compressor measured by Power Meter (Wh)
    # V: Volume of refrigerant measured by totalizer (L)
    
    
    # Import Radial Test Data from experiment
    refrigData = pd.read_csv(file, sep='\t', header=21)

    # Create time column with time step and index
    refrigData['Time (s)'] = refrigData['X_Value'] * \
        refrigData['Time Step (s)'][0]

    # Create pressure columns with transducer fit function
    refrigData['Pressure Transducer 1 (PaG)'] = pressureFit(
        refrigData['Pressure  1'])
    refrigData['Pressure Transducer 2 (PaG)'] = pressureFit(
        refrigData['Pressure 2'])
    refrigData['Pressure Transducer 3 (PaG)'] = pressureFit(
        refrigData['Pressure 3'])
    refrigData['Pressure Transducer 4 (PaG)'] = pressureFit(
        refrigData['Pressure 4'])

    # Determine index to access last ten minutes of test 600s = 10 min
    lastTen = int(600 / refrigData['Time Step (s)'][0])
    
    # Calculate mean ambient temperature
    T_amb = refrigData['Thermistor (C)'][-lastTen:-1].mean()  + 273.15
    
    # Compute mean outlet air temperature from condenser
    try:
        T_o_air = refrigData['Condenser Exit Temp'][-lastTen:-1].mean()  + 273.15
    except:
        T_o_air = None
    # Compute air speed measured behind condenser (m/s) [not very accurate]
    windSpeed = np.sqrt(refrigData['Pitot Tube (Torr)'].abs(
    ) * 133.322 * 2 / CP.PropsSI('D', 'T', T_amb, 'P', P_amb, 'air')).mean()

    # Compute mean steady state radial temperatures from last 600 seconds
    radialProfile = refrigData[['Thermocouple 2 (C)', 'Thermocouple 4 (C) ', 
                                'Thermocouple 7 (C)']][-lastTen:-1].mean() + offset    
    # Thermocouple location with origin at center of heating element
    locations = np.array([1.5, 7.5, 16.5])

    # Compute area for each thermocouple
    dr = locations / 100 - np.array([0, locations[0], locations[1]]) / 100

    # compute mean internal temperature
    T_mean_pod = np.dot(radialProfile, dr) / (locations [2] / 100)


    # Calculate steady state mean pressures and temperatures
    pressures = refrigData[['Pressure Transducer 1 (PaG)', 'Pressure Transducer 3 (PaG)', 
                            'Pressure Transducer 2 (PaG)',
                            'Pressure Transducer 4 (PaG)']][-lastTen:-1].mean().values + P_amb  # Pressure from barometer

    # Put in proper order: post evap, post compressor, post condenser, post capillary tubes
    pressures = pressures[[1, 2, 3, 0]]

    temperatures = refrigData[['Temperature  1', 'Temperature 2', 'Temperature 3', 'Temperature 4'
                               ]][-lastTen:-1].mean().values + 273.15 + offset

    # Put in proper order: post evap, post compressor, post condenser, post capillary tubes
    temperatures = temperatures[[1, 2, 3, 0]]
    
    # Calculate compensation due to temperature of refrigerant being measured on the outside surface of the tubing
    # http://www.burnsengineering.com/local/uploads/files/small_diameter_lines.pdf
    temperatures[0:3] = temperatures[0:3] + 0.02 * (temperatures[0:3] - T_amb)

    temperatures[3] = temperatures[3] + 0.02 * (temperatures[3] - T_mean_pod - 273.15)
    
    # Look up saturated enthalpy and entropy
    refrigerant = 'R410a'

    #Compute Enthalpy and entropy from P and T
    cycleEnthalpy = CP.PropsSI('H', 'P', pressures, 'T', temperatures, refrigerant)

    cycleEntropy = CP.PropsSI('S', 'P', pressures, 'T', temperatures, refrigerant)

    # Check if any positions are two phase:
    TP = np.isinf(cycleEnthalpy)
    
    for ind, point in enumerate(TP):
        if point:
            x = (pressures[ind] - CP.PropsSI('P', 'T', temperatures[ind], 'Q', 1, refrigerant)) / \
            (CP.PropsSI('P', 'T', temperatures[ind], 'Q', 0, refrigerant) - 
             CP.PropsSI('P', 'T', temperatures[ind], 'Q', 1, refrigerant))

            cycleEnthalpy[ind] = CP.PropsSI('H', 'P', pressures[ind], 'Q', x, refrigerant)
            cycleEntropy[ind] = CP.PropsSI('S', 'P', pressures[ind], 'Q', x, refrigerant)    
    
    # Convert Work of refrigeration system from Wh to W divide by number of samples (W)
    W_refrig = W_refrig / refrigData['Time (s)'].tail(1) * 60 * 60 
    
    # Compute compressor by subrtracting blower work. 
    # Blower work from test on 01/23 0.08 KWh after 60 min.  (W)
    W_comp = W_refrig - 0.08 * 1000 
    
    if not np.isnan(V):
        
        # Compute Density
        rho = CP.PropsSI('D', 'T', temperatures[0], 'P', pressures[0], refrigerant)
        # Compute mass flowrate of refrigerant
        m_dot = V / refrigData['Time (s)'].tail(1) * rho * 0.001
        
    else:
        m_dot = np.nan
    
    # Convert power of heat element from Wh to W divide by number of samples
    Q_element = Q_element / refrigData['Time (s)'].tail(1) * 60 * 60 
    
    # Compute Evap heat load (W)
    Q_L = m_dot * (cycleEnthalpy[0] - cycleEnthalpy[2])
    
    # Compute Cond. heat load (W)
    Q_H = m_dot * (cycleEnthalpy[1] - cycleEnthalpy[2])
    
    # Compute ambient heat load (W)
    Q_ambient = Q_element - Q_L
    
    # COSP (Condesner fan work from test on 01/23 0.04 Kwh after 60 mins) 
    COSP = Q_L / (W_refrig + 0.04 * 1000)
    
    # COP 
    COP = Q_L / W_comp
    
    # Creae pandas dataframe for 
    experimentalData = pd.DataFrame({'Ambient P (Pa)': P_amb, 
                                     'Ambient T (K)': T_amb, 
                                     'P (Pa)': [pressures], 
                                     'T (K)': [temperatures], 
                                     'h (j/kg)':[cycleEnthalpy], 
                                     's (j/kg K)': [cycleEntropy], 
                                     'Mass Flux (kg/s)': m_dot,
                                     'Pod T Profile (K)': [radialProfile.values], 
                                     'Pod T (K)': [T_mean_pod + 273.15],
                                     'Delta T (K)' : (T_mean_pod + 273.15) - T_amb,
                                     'Air T Condenser Exit (K)': windSpeed,
                                     'Air Speed Condenser (m/s)': T_o_air,
                                     'Heating Element Power (W)': Q_element, 
                                     'Ambient Heat Load (W)': Q_ambient, 
                                     'Q_L (W)': Q_L, 
                                     'Q_H (W)': Q_H,
                                     'Compressor Work (W)': W_comp,
                                     'COSP': COSP,
                                     'COP': COP,
                                     'Thermocouple offset T (K)': offset,
                                     'Raw Data': [refrigData],
                                     'file': file,})

    

    return experimentalData


def experimental_analysis_cycle(file, P_amb, Q_element, W_refrig, offset = 2):
    # A function which performs analysis of temperature measurements  
    # taken during the prototype compressor cycling experiments and
    # returns mean values for temperatures, work and heat transfer
    
    # file: Location of experimental data file.
    # P_amb: Ambient pressure measured by altimeter. (Pa)
    # Q_element: Input Power to heating element. measured by Power Meter (Wh)
    # W_refrig: Power consumed by blower and compressor measured by Power Meter (Wh)
    
    
    # Import Radial Test Data from experiment
    refrigData = pd.read_csv(file, sep='\t', header=21)

    # Create time column with time step and index
    refrigData['Time (s)'] = refrigData['X_Value'] * \
        refrigData['Time Step (s)'][0]

    # Create pressure columns with transducer fit function
    refrigData['Pressure Transducer 1 (PaG)'] = pressureFit(
        refrigData['Pressure  1'])
    refrigData['Pressure Transducer 2 (PaG)'] = pressureFit(
        refrigData['Pressure 2'])
    refrigData['Pressure Transducer 3 (PaG)'] = pressureFit(
        refrigData['Pressure 3'])
    refrigData['Pressure Transducer 4 (PaG)'] = pressureFit(
        refrigData['Pressure 4'])
    
    # Calculate mean ambient temperature
    T_amb = refrigData['Thermistor (C)'].mean()  + 273.15

    # Compute mean steady state radial temperatures from last 600 seconds
    radialProfile = refrigData[['Thermocouple 2 (C)', 'Thermocouple 4 (C) ', 
                                'Thermocouple 7 (C)']].mean() + offset    
    # Thermocouple location with origin at center of heating element
    locations = np.array([1.5, 7.5, 16.5])

    # Compute area for each thermocouple
    dr = locations / 100 - np.array([0, locations[0], locations[1]]) / 100

    # compute mean internal temperature
    T_mean_pod = np.dot(radialProfile, dr) / (locations [2] / 100)    
    
    # Convert Work of refrigeration system from Wh to W divide by number of samples (W)
    W_refrig = W_refrig / refrigData['Time (s)'].tail(1) * 60 * 60 
    
    # Compute compressor by subrtracting blower work. 
    # Blower work from test on 01/23 0.08 KWh after 60 min.  (W)
    W_comp = W_refrig - 0.08 * 1000 
    
    # Convert power of heat element from Wh to W divide by number of samples
    Q_element = Q_element / refrigData['Time (s)'].tail(1) * 60 * 60 
    
    # Compute ambient heat load based on best fit from test analysis (W)
    Q_ambient = ((T_mean_pod + 273.15) - T_amb) * 24.565532583275
    
    # Compute Evap heat load (W)
    Q_L = Q_element - Q_ambient
    
    # COSP (Condesner fan work from test on 01/23 0.04 Kwh after 60 mins) 
    COSP = Q_L / (W_refrig + 0.04 * 1000)
    
    # COP 
    COP = Q_L / W_comp
    
    # Creae pandas dataframe for 
    experimentalData = pd.DataFrame({'Ambient P (Pa)': P_amb, 
                                     'Ambient T (K)': T_amb, 
                                     'Pod T Profile (K)': [radialProfile.values], 
                                     'Pod T (K)': [T_mean_pod + 273.15],
                                     'Delta T (K)' : (T_mean_pod + 273.15) - T_amb,
                                     'Heating Element Power (W)': Q_element, 
                                     'Ambient Heat Load (W)': Q_ambient, 
                                     'Q_L (W)': Q_L,
                                     'Compressor Work (W)': W_comp,
                                     'COSP': COSP,
                                     'COP': COP,
                                     'Thermocouple offset T (K)': offset,
                                     'Raw Data': [refrigData],
                                     'file': file,})
    return experimentalData


#     # Model pod as rectangular prism
#     # Incropera, Frank P.; DeWitt, David P. (2007). 
#     # Fundamentals of Heat and Mass Transfer (6th ed.). 
#     # Hoboken: Wiley. pp. 578, ISBN 978-0-471-45728-2.
    
#     # Length of prototype (m) [86.5"]
#     L = 86.5 * 0.0254
    
#     # width of prototype (m) [20"]
#     W = (20) * 0.0254
    
#     # Height of prototype body (m) [20"]
#     H = W
    
#     # Thickness of plywood (m) [0.5"]
#     t = 0.5 / 16 * 0.0254
    
#     # thermal conductivity of plywood (W/mk)
#     k_ply = 0.13
    
#     # film temperature (K)
#     T_f = (T_mean_pod + 273.15 + T_amb) / 2
    
#     # Temperature difference between ambient and pod (K)
#     delta_T = np.abs(T_mean_pod + 273.15 - T_amb)
    
#     # Thermal conductivity (W/m K)
#     k  = CP.PropsSI('L', 'T', T_f, 'P', P_amb, 'air')
    
#     # isobaric Specific Heat of air (j/kg/k)
#     C_p = CP.PropsSI('C', 'T', T_f, 'P', P_amb, 'air')
    
#     # absolute viscosity of air (Pa*s)
#     mu = CP.PropsSI('V', 'T', T_f, 'P', P_amb, 'air')
    
#     # Density of air (kg/m^3) 
#     rho = CP.PropsSI('D', 'T', T_f, 'P', P_amb, 'Air')
    
#     # Kinematic Viscosity (m^2/s)
#     nu = mu / rho
    
#     # Prandtl number of air
#     Pr = CP.PropsSI('Prandtl', 'T', T_f, 'P', P_amb, 'Air')
    
#     # Isobaric expansion coefficient (1/K)
#     beta = CP.PropsSI('isobaric_expansion_coefficient', 'T', T_f, 'P', P_amb, 'Air')
    
#     # gravity (m/s^2)
#     g = 9.81
    
#     # Thermal diffusivity (m^2/s)
#     alpha = k / rho / C_p
    
#     # Raleigh Number sides
#     Ra_s = g * beta * delta_T * (H)**3 / nu / alpha
    
#     # Raleigh Number top 
#     Ra_t = g * beta * delta_T * (W / 2)**3 / nu / alpha
    
#     # Raleigh Number bottom (same as top)
#     Ra_b = g * beta * delta_T * (W / 2)**3 / nu / alpha
    
    
    
#     # Compute Nusselt number
#     if Ra_s < 1e9:
#         Nu_s = (0.68 + 0.67 * Ra_s**(1/4) / (1 + (0.492 / Pr)**(9/16))**(4/9))
#     else:
#         raise ValueError('Ra sides is not within range ' + str(Ra_s))
        
#     if Ra_t < 1e11 and Ra_t > 1e7:
#         Nu_t = 0.15 * Ra_t**(1/3)
#     elif Ra_t < 1e7 and Ra_t > 1e4:
#         Nu_t = 0.54 * Ra_t**(1/4)
#     else:
#         raise ValueError('Ra top is not within range ' + str(Ra_t))  
        
#     if Ra_b < 1e10 and Ra_b > 1e5:
#         Nu_b = 0.27 * Ra_b**(1/4)
#     else:
#         raise ValueError('Ra bottom is not within range ' + str(Ra_b))

    
#     # convection coefficient
#     h_s =  Nu_s * k / (H)
#     h_t =  Nu_t * k / (W / 2)
#     h_b =  Nu_b * k / (W / 2)
    
#     # Ambient heat load (W)
#     Q_ambient = delta_T * (2 * H * L / (t / k_ply + 1 / h_s) + # Sides
#                                                        2 * H * W / (t / k_ply + 1 / h_s) + # front and back
#                                                        W * L / (t / k_ply + 1 / h_t) + # top
#                                                        W * L / (t / k_ply + 1 / h_b)) # bottom


    # Modeled as Cylinder
#     # Inner Radius
#     r_i = 7.25 * 0.0254
    
#     # Outer Radius
#     r_o = (r_i + t) * 0.0254
    
#     Ra = g * beta * delta_T * (2 * r_o)**3 / nu / alpha
    
#     # Compute Nusselt number
#     if Ra < 1e11:
#         Nu = (0.60 + 0.387 * Ra**(1/6) / (1 + (0.559 / Pr)**(9/16))**(8/27))**2
#     else:
#         raise ValueError('Ra sides is not within range ' + str(Ra))
        
#     # Compute Convection Coef.
#     h = Nu * k / (2 * r_o)
    
#     # Ambient heat load (W)
#     Q_ambient = (radialProfile[2] + 273.15 - T_amb) / (1 / ((2 * r_o) * np.pi * L * h) + 
#                                                            np.log(r_o / r_i) / (2 * k_ply * np.pi * L))


def thermodynamic_plots(*args, lgnd = ['Vapor Dome', 'Ambient Temperature', 'Pod Temperature'], 
                        annotate = False, style = ["go", "-o"], save = False):
    # A function which plots the T-s and P-h diagram of the experimental 
    # measurements of the VCRC and model of the VCRC. If only one arguement is given
    # the function assumes experimental data and plots T-s and P-h numbered points.
    
    # arg 1: Expereimental data in pandas dataframe
    # arg 2: Data to compare in pandas dataframe
    txt = ["1'", "2'", "5'", "6'"]
    
    if len(args) == 2:
        exp = args[0] 
        model = args[1]
        refrigerant = 'R410a'
        # Create Vapor Dome
        vaporDomeS = np.concatenate([CP.PropsSI('S', 'T', np.linspace(200, 344, 150), 'Q', 0, refrigerant),
                                     CP.PropsSI('S', 'T', np.linspace(344, 200, 150), 'Q', 1, refrigerant)])
        vaporDomeT = np.concatenate(
            [np.linspace(200, 344, 150), np.linspace(344, 200, 150)])

        vaporDomeH = np.concatenate([CP.PropsSI('H', 'P', np.linspace(2e5, 5000e3, 150), 'Q', 0, refrigerant),
                                     CP.PropsSI('H', 'P', np.linspace(5000e3, 2e5, 150), 'Q', 1, refrigerant)])
        vaporDomeP = np.concatenate(
            [np.linspace(2e5, 5000e3, 150), np.linspace(5000e3, 2e5, 150)])


        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        plt.subplot(121)
        plt.plot(vaporDomeS, vaporDomeT, '-')
        plt.plot(model['s (j/kg K)'], model['T (K)'], style[1])
        plt.plot(exp['s (j/kg K)'], exp['T (K)'], style[0])
        if annotate:
            for ind, s in enumerate(exp['s (j/kg K)']):
                plt.annotate(txt[ind], (exp['s (j/kg K)'][ind] + 10, exp['T (K)'][ind] + 1))
            
        plt.plot([500, 2000],[exp['Ambient T (K)'], exp['Ambient T (K)']], 'c--')
        plt.plot([500, 2000],[exp['Pod T (K)'], exp['Pod T (K)']], 'r--')
        plt.ylabel('Temperature (K)')
        plt.xlabel('Entropy (j/kg/K)')
        plt.legend(lgnd)


        plt.title('Prototype Refrigeration Sys. T-s ')
        plt.ylim(exp['T (K)'].min() - 10)

        plt.subplot(122)
        plt.plot(vaporDomeH, vaporDomeP, '-')
        plt.plot(model['h (j/kg)'], model['P (Pa)'], style[1])
        plt.plot(exp['h (j/kg)'], exp['P (Pa)'], style[0])
        if annotate:
            for ind, h in enumerate(exp['h (j/kg)']):
                plt.annotate(txt[ind], (exp['h (j/kg)'][ind] + 1e3, exp['P (Pa)'][ind] + 0.1e5))
            
        plt.ylabel('Pressure (Pa)')
        plt.xlabel('Enthalpy (j/kg)')
        plt.ylim(exp['P (Pa)'].min() - 500e3)
        plt.title('Prototype Refrigeration Sys. P-h ')
        
    if len(args) == 1:
        exp = args[0] 
        refrigerant = 'R410a'
        # Create Vapor Dome
        vaporDomeS = np.concatenate([CP.PropsSI('S', 'T', np.linspace(200, 344, 150), 'Q', 0, refrigerant),
                                     CP.PropsSI('S', 'T', np.linspace(344, 200, 150), 'Q', 1, refrigerant)])
        vaporDomeT = np.concatenate(
            [np.linspace(200, 344, 150), np.linspace(344, 200, 150)])

        vaporDomeH = np.concatenate([CP.PropsSI('H', 'P', np.linspace(2e5, 5000e3, 150), 'Q', 0, refrigerant),
                                     CP.PropsSI('H', 'P', np.linspace(5000e3, 2e5, 150), 'Q', 1, refrigerant)])
        vaporDomeP = np.concatenate(
            [np.linspace(2e5, 5000e3, 150), np.linspace(5000e3, 2e5, 150)])

        
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        plt.subplot(121)
        plt.plot(vaporDomeS, vaporDomeT, '-')
        plt.plot([500, 2000],[exp['Ambient T (K)'], exp['Ambient T (K)']], 'c--')
        plt.plot([500, 2000],[exp['Pod T (K)'], exp['Pod T (K)']], 'r--')
        
        if annotate:
            for ind, s in enumerate(exp['s (j/kg K)']):
                plt.annotate(txt[ind], (exp['s (j/kg K)'][ind] + 10, exp['T (K)'][ind] + 1))
            
        plt.plot(exp['s (j/kg K)'], exp['T (K)'], style[0])
        plt.ylabel('Temperature (K)')
        plt.xlabel('Entropy (j/kg/K)')
        plt.legend(lgnd)


        plt.title('Prototype Refrigeration Sys. T-s ')
        plt.ylim(exp['T (K)'].min() - 10)
        plt.subplot(122)
        plt.plot(vaporDomeH, vaporDomeP, '-')
        plt.plot(exp['h (j/kg)'], exp['P (Pa)'], style[0])
        
        if annotate:
            for ind, h in enumerate(exp['h (j/kg)']):
                plt.annotate(txt[ind], (exp['h (j/kg)'][ind] + 1e3, exp['P (Pa)'][ind] + 0.1e5))
            
        plt.ylabel('Pressure (Pa)')
        plt.xlabel('Enthalpy (j/kg)')
        plt.ylim(exp['P (Pa)'].min() - 500e3)
        plt.title('Prototype Refrigeration Sys. P-h ')
    
    if save:
        plt.savefig(str(exp.name) + ".png")
    else:
        plt.show()
        
    return

def example_plots1(save = False):
    # A function which plots example T-s and P-h diagrams of a VCRC with and without losses.
    
    refrigerant = 'R410a' 
    
    P_e = 1e6
    P_c = 3.5e6

    # Create Vapor Dome
    vaporDomeS = np.concatenate([CP.PropsSI('S', 'T', np.linspace(200, 344, 150), 'Q', 0, refrigerant),
                                 CP.PropsSI('S', 'T', np.linspace(344, 200, 150), 'Q', 1, refrigerant)])
    vaporDomeT = np.concatenate(
        [np.linspace(200, 344, 150), np.linspace(344, 200, 150)])

    vaporDomeH = np.concatenate([CP.PropsSI('H', 'P', np.linspace(2e5, 5000e3, 150), 'Q', 0, refrigerant),
                                 CP.PropsSI('H', 'P', np.linspace(5000e3, 2e5, 150), 'Q', 1, refrigerant)])
    vaporDomeP = np.concatenate(
        [np.linspace(2e5, 5000e3, 150), np.linspace(5000e3, 2e5, 150)])

    # Starting evaporation and condensing pressures
    idealPressure = np.array(P_e)
    idealPressure = np.append(idealPressure, P_c * np.ones(4))
    idealPressure = np.append(idealPressure, P_e * np.ones(3))


    # Starting temperature with 15 degree super heat
    idealTemperature = np.array(CP.PropsSI('T', 'P', idealPressure[0], 'Q', 1, refrigerant) + 15)

    # Look up entropy
    idealEntropy = np.array(CP.PropsSI('S', 'P', idealPressure[0], 'T', idealTemperature.flat[0], refrigerant))

    # Look up enthalpy
    idealEnthalpy = np.array(CP.PropsSI('H', 'P', idealPressure[0], 'T', idealTemperature.flat[0], refrigerant))

    # Isentropic Compression
    idealEntropy = np.append(idealEntropy, idealEntropy.flat[0])
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[1], 'S', idealEntropy[1], refrigerant))
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[1], 'S', idealEntropy[1], refrigerant))

    # Isobaric super heat
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[2], 'Q', 1, refrigerant))
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[2], 'Q', 1, refrigerant))
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[2], 'Q', 1, refrigerant))

    # Isobaric phase change
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[3], 'Q', 0, refrigerant))
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[3], 'Q', 0, refrigerant))
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[3], 'Q', 0, refrigerant))

    # 10 degree subcool
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[4], 'Q', 0, refrigerant) -  10)
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[4], 'T', idealTemperature[4], refrigerant))
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[4], 'T', idealTemperature[4], refrigerant))

    # Isenthalpic expansion
    idealEnthalpy = np.append(idealEnthalpy, idealEnthalpy[4])
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[5], 'H', idealEnthalpy[5], refrigerant))
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[5], 'H', idealEnthalpy[5], refrigerant))


    # Isobaric phase change
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[6], 'Q', 1, refrigerant))
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[6], 'Q', 1, refrigerant))
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[6], 'Q', 1, refrigerant))

    # Isobaric superheat 
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[7], 'T', idealTemperature[0], refrigerant))
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[7], 'T', idealTemperature[0], refrigerant))
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[7], 'T', idealTemperature[0], refrigerant))



    insideTemp = 300
    outsideTemp = 310
    

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(2.4*4, 1*4))
    ax = plt.subplot(121)
    plt.plot(vaporDomeS, vaporDomeT, '--k')
    txt = list(range(1, 7)) + list(range(8,9))

    for ind, s in enumerate(idealEntropy):
        if ind == len(idealEnthalpy) - 1:
            break
        plt.annotate(str(txt[ind]), (idealEntropy[ind] + 10, idealTemperature[ind] + 1))
    plt.plot(idealEntropy, idealTemperature, 'o-k')

    plt.plot([500, 2000],[outsideTemp, outsideTemp], 'k.-')
    plt.plot([500, 2000],[insideTemp, insideTemp], 'k..')
    plt.ylabel('Temperature')
    plt.xlabel('Entropy')
    plt.legend(['Vapor Dome', 'Ideal Cycle', 'Ambient Temperature', 'Intertior Temperature'])
    plt.title('T-s ')
    plt.ylim(idealTemperature.min() - 10)
    
    plt.tick_params(axis = "both", which = "both", length = 0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    ax = plt.subplot(122)
    plt.plot(vaporDomeH, vaporDomeP, '-')

    plt.plot(idealEnthalpy, idealPressure, '-o')
    for ind, h in enumerate(idealEnthalpy):
        if ind == len(idealEnthalpy) - 1:
            break
        plt.annotate(str(txt[ind]), (idealEnthalpy[ind] + 3 * 1e3, idealPressure[ind] + 3 * 0.1e5))

    plt.ylabel('Pressure')
    plt.xlabel('Enthalpy')
    plt.ylim(idealPressure.min() - 500e3)
    plt.title('P-h ')

    plt.tick_params(axis = "both", which = "both", length = 0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    if save:
        plt.savefig("T-s P-h.png")
    else:
        plt.show()

    return

def example_plots(save = False):
    # A function which plots example T-s and P-h diagrams of a VCRC with and without losses.
    
    refrigerant = 'R410a' 
    
    P_e = 1e6
    P_c = 3.5e6

    # Create Vapor Dome
    vaporDomeS = np.concatenate([CP.PropsSI('S', 'T', np.linspace(200, 344, 150), 'Q', 0, refrigerant),
                                 CP.PropsSI('S', 'T', np.linspace(344, 200, 150), 'Q', 1, refrigerant)])
    vaporDomeT = np.concatenate(
        [np.linspace(200, 344, 150), np.linspace(344, 200, 150)])

    vaporDomeH = np.concatenate([CP.PropsSI('H', 'P', np.linspace(2e5, 5000e3, 150), 'Q', 0, refrigerant),
                                 CP.PropsSI('H', 'P', np.linspace(5000e3, 2e5, 150), 'Q', 1, refrigerant)])
    vaporDomeP = np.concatenate(
        [np.linspace(2e5, 5000e3, 150), np.linspace(5000e3, 2e5, 150)])

    # Starting evaporation and condensing pressures
    idealPressure = np.array(P_e)
    idealPressure = np.append(idealPressure, P_c * np.ones(4))
    idealPressure = np.append(idealPressure, P_e * np.ones(3))


    # Starting temperature with 15 degree super heat
    idealTemperature = np.array(CP.PropsSI('T', 'P', idealPressure[0], 'Q', 1, refrigerant) + 15)

    # Look up entropy
    idealEntropy = np.array(CP.PropsSI('S', 'P', idealPressure[0], 'T', idealTemperature.flat[0], refrigerant))

    # Look up enthalpy
    idealEnthalpy = np.array(CP.PropsSI('H', 'P', idealPressure[0], 'T', idealTemperature.flat[0], refrigerant))

    # Isentropic Compression
    idealEntropy = np.append(idealEntropy, idealEntropy.flat[0])
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[1], 'S', idealEntropy[1], refrigerant))
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[1], 'S', idealEntropy[1], refrigerant))

    # Isobaric super heat
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[2], 'Q', 1, refrigerant))
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[2], 'Q', 1, refrigerant))
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[2], 'Q', 1, refrigerant))

    # Isobaric phase change
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[3], 'Q', 0, refrigerant))
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[3], 'Q', 0, refrigerant))
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[3], 'Q', 0, refrigerant))

    # 10 degree subcool
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[4], 'Q', 0, refrigerant) -  10)
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[4], 'T', idealTemperature[4], refrigerant))
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[4], 'T', idealTemperature[4], refrigerant))

    # Isenthalpic expansion
    idealEnthalpy = np.append(idealEnthalpy, idealEnthalpy[4])
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[5], 'H', idealEnthalpy[5], refrigerant))
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[5], 'H', idealEnthalpy[5], refrigerant))


    # Isobaric phase change
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[6], 'Q', 1, refrigerant))
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[6], 'Q', 1, refrigerant))
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[6], 'Q', 1, refrigerant))

    # Isobaric superheat 
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[7], 'T', idealTemperature[0], refrigerant))
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[7], 'T', idealTemperature[0], refrigerant))
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[7], 'T', idealTemperature[0], refrigerant))


    ########################### REAL Cycle ###########################

    # Starting evaporation and condensing pressures
    P_drop = 1e5
    
    actualPressure = np.array(P_e - P_drop)
    actualPressure = np.append(actualPressure, P_c)


    # Starting temperature with 15 degree super heat
    actualTemperature = np.array(CP.PropsSI('T', 'P', actualPressure[0], 'Q', 1, refrigerant) + 15)

    # Look up entropy
    actualEntropy = np.array(CP.PropsSI('S', 'P', actualPressure[0], 'T', actualTemperature.flat[0], refrigerant))

    # Look up enthalpy
    actualEnthalpy = np.array(CP.PropsSI('H', 'P', actualPressure[0], 'T', actualTemperature.flat[0], refrigerant))

    # Compression with 50% superheat entropy loss
    actualEntropy = np.append(actualEntropy, actualEntropy.flat[0] - 0.5 * (actualEntropy.flat[0] - 
                                                                            CP.PropsSI('S', 'P', actualPressure[1], 'Q', 1, refrigerant)))
    actualTemperature = np.append(actualTemperature, CP.PropsSI('T', 'P', actualPressure[1], 'S', actualEntropy[1], refrigerant))
    actualEnthalpy = np.append(actualEnthalpy, CP.PropsSI('H', 'P', actualPressure[1], 'S', actualEntropy[1], refrigerant))

    # super heat
    actualPressure = np.append(actualPressure, actualPressure[1] - 0.25 * P_drop)
    actualEntropy = np.append(actualEntropy, CP.PropsSI('S', 'P', actualPressure[2], 'Q', 1, refrigerant))
    actualTemperature = np.append(actualTemperature, CP.PropsSI('T', 'P', actualPressure[2], 'Q', 1, refrigerant))
    actualEnthalpy = np.append(actualEnthalpy, CP.PropsSI('H', 'P', actualPressure[2], 'Q', 1, refrigerant))

    # phase change
    actualPressure = np.append(actualPressure, actualPressure[1] - 0.5 * P_drop)
    actualEntropy = np.append(actualEntropy, CP.PropsSI('S', 'P', actualPressure[3], 'Q', 0, refrigerant))
    actualTemperature = np.append(actualTemperature, CP.PropsSI('T', 'P', actualPressure[3], 'Q', 0, refrigerant))
    actualEnthalpy = np.append(actualEnthalpy, CP.PropsSI('H', 'P', actualPressure[3], 'Q', 0, refrigerant))

    # 10 degree subcool
    actualPressure = np.append(actualPressure, actualPressure[1] - 1e5)
    actualTemperature = np.append(actualTemperature, CP.PropsSI('T', 'P', actualPressure[4], 'Q', 0, refrigerant) -  10)
    actualEntropy = np.append(actualEntropy, CP.PropsSI('S', 'P', actualPressure[4], 'T', actualTemperature[4], refrigerant))
    actualEnthalpy = np.append(actualEnthalpy, CP.PropsSI('H', 'P', actualPressure[4], 'T', actualTemperature[4], refrigerant))

    # Isenenthalpic expansion
    actualPressure = np.append(actualPressure, actualPressure[0] + P_drop)
    actualEnthalpy = np.append(actualEnthalpy, 1 * actualEnthalpy[4])
    actualTemperature = np.append(actualTemperature, CP.PropsSI('T', 'P', actualPressure[5], 'H', actualEnthalpy[5], refrigerant))
    actualEntropy = np.append(actualEntropy, CP.PropsSI('S', 'P', actualPressure[5], 'H', actualEnthalpy[5], refrigerant))


    # phase change
    actualPressure = np.append(actualPressure, actualPressure[0] + 0.25 * P_drop)
    actualEnthalpy = np.append(actualEnthalpy, CP.PropsSI('H', 'P', actualPressure[6], 'Q', 1, refrigerant))
    actualTemperature = np.append(actualTemperature, CP.PropsSI('T', 'P', actualPressure[6], 'Q', 1, refrigerant))
    actualEntropy = np.append(actualEntropy, CP.PropsSI('S', 'P', actualPressure[6], 'Q', 1, refrigerant))

    # superheat 
    actualPressure = np.append(actualPressure, actualPressure[0])
    actualEnthalpy = np.append(actualEnthalpy, CP.PropsSI('H', 'P', actualPressure[7], 'T', actualTemperature[0], refrigerant))
    actualTemperature = np.append(actualTemperature, CP.PropsSI('T', 'P', actualPressure[7], 'T', actualTemperature[0], refrigerant))
    actualEntropy = np.append(actualEntropy, CP.PropsSI('S', 'P', actualPressure[7], 'T', actualTemperature[0], refrigerant))

    insideTemp = 300
    outsideTemp = 310
    

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    plt.subplot(121)
    plt.plot(vaporDomeS, vaporDomeT, '-')
    txt = list(range(1, 7)) + list(range(8,9))

    for ind, s in enumerate(idealEntropy):
        if ind == len(idealEnthalpy) - 1:
            break
        plt.annotate(str(txt[ind]), (idealEntropy[ind] + 10, idealTemperature[ind] + 1))
    plt.plot(idealEntropy, idealTemperature, 'o-')

    for ind, s in enumerate(actualEntropy):
        if ind == len(actualEnthalpy) - 1:
            break
        plt.annotate(str(txt[ind]) + "'", (actualEntropy[ind] - 5 * 10, actualTemperature[ind] + 2))
    plt.plot(actualEntropy, actualTemperature, 'og--')
    plt.plot([500, 2000],[outsideTemp, outsideTemp], 'r--')
    plt.plot([500, 2000],[insideTemp, insideTemp], 'c--')
    plt.ylabel('Temperature (K)')
    plt.xlabel('Entropy (j/kg/K)')
    plt.legend(['Vapor Dome', 'Ideal Cycle', 'Cycle with Losses', 'Outside Temperature', 'Inside Temperature'])
    plt.title('Prototype Refrigeration Sys. T-s ')
    plt.ylim(actualTemperature.min() - 10)
    
    plt.subplot(122)
    plt.plot(vaporDomeH, vaporDomeP, '-')

    plt.plot(idealEnthalpy, idealPressure, '-o')
    for ind, h in enumerate(idealEnthalpy):
        if ind == len(idealEnthalpy) - 1:
            break
        plt.annotate(str(txt[ind]), (idealEnthalpy[ind] + 3 * 1e3, idealPressure[ind] + 3 * 0.1e5))

    plt.plot(actualEnthalpy, actualPressure, '--go')
    for ind, h in enumerate(actualEnthalpy):
        if ind == len(actualEnthalpy) - 1:
            break
        plt.annotate(str(txt[ind]) + "'", (actualEnthalpy[ind] - 1e4, actualPressure[ind] - 2 * 1e5))

    plt.ylabel('Pressure (Pa)')
    plt.xlabel('Enthalpy (j/kg)')
    plt.ylim(actualPressure.min() - 500e3)
    plt.title('Prototype Refrigeration Sys. P-h ')

    plt.show() 

    return

def ideal_cycle(P_c, P_e, T_SC, T_SH):
    # A function which plots example T-s and P-h diagrams of a VCRC with and without losses.
    
    refrigerant = 'R410a' 

    # Starting evaporation and condensing pressures
    idealPressure = np.array(P_e)
    idealPressure = np.append(idealPressure, P_c * np.ones(4))
    idealPressure = np.append(idealPressure, P_e * np.ones(3))


    # Starting temperature with 15 degree super heat
    idealTemperature = np.array(CP.PropsSI('T', 'P', idealPressure[0], 'Q', 0, refrigerant) + T_SH)

    # Look up entropy
    idealEntropy = np.array(CP.PropsSI('S', 'P', idealPressure[0], 'T', idealTemperature.flat[0], refrigerant))

    # Look up enthalpy
    idealEnthalpy = np.array(CP.PropsSI('H', 'P', idealPressure[0], 'T', idealTemperature.flat[0], refrigerant))

    # Isentropic Compression
    idealEntropy = np.append(idealEntropy, idealEntropy.flat[0])
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[1], 'S', idealEntropy[1], refrigerant))
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[1], 'S', idealEntropy[1], refrigerant))

    # Isobaric super heat
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[2], 'Q', 1, refrigerant))
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[2], 'Q', 1, refrigerant))
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[2], 'Q', 1, refrigerant))

    # Isobaric phase change
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[3], 'Q', 0, refrigerant))
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[3], 'Q', 0, refrigerant))
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[3], 'Q', 0, refrigerant))

    # 10 degree subcool
    idealTemperature = np.append(idealTemperature, idealTemperature[-1] - T_SC)
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[4], 'T', idealTemperature[4], refrigerant))
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[4], 'T', idealTemperature[4], refrigerant))

    # Isenthalpic expansion
    idealEnthalpy = np.append(idealEnthalpy, idealEnthalpy[4])
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[5], 'H', idealEnthalpy[5], refrigerant))
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[5], 'H', idealEnthalpy[5], refrigerant))


    # Isobaric phase change
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[6], 'Q', 1, refrigerant))
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[6], 'Q', 1, refrigerant))
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[6], 'Q', 1, refrigerant))

    # Isobaric superheat 
    idealEnthalpy = np.append(idealEnthalpy, CP.PropsSI('H', 'P', idealPressure[7], 'T', idealTemperature[0], refrigerant))
    idealTemperature = np.append(idealTemperature, CP.PropsSI('T', 'P', idealPressure[7], 'T', idealTemperature[0], refrigerant))
    idealEntropy = np.append(idealEntropy, CP.PropsSI('S', 'P', idealPressure[7], 'T', idealTemperature[0], refrigerant))

    # Creae pandas dataframe for 
    idealData = pd.DataFrame({ 'P (Pa)': [idealPressure], 'T (K)': [idealTemperature], 'h (j/kg)':[idealEnthalpy], 
                                     's (j/kg K)': [idealEntropy]})
    
    return idealData