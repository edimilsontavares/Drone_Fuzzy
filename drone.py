# -*- coding: utf-8 -*-
"""

@author: grupp
"""

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0
    def compute(self, setpoint, actual_value, dt):
        error = setpoint - actual_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative
    

class DroneAltitude:
    def __init__(self, m, Cd):
        self.altitude = 0
        self.velocidade = 0
        self.m = m
        self.Cd = Cd
    def update(self, impulso, dt):
        g = 9.81
        impulso = np.clip(impulso, 0, 100)
        aceleracao = (impulso - self.Cd * self.velocidade - self.m * g) / self.m
        self.velocidade += aceleracao * dt
        self.altitude += self.velocidade * dt
        return self.altitude
    
# Funções para cálculo das métricas
def calculate_metrics(tempo, altitudes, setpoint):
    altitudes = np.array(altitudes) 
    # Erro Médio Absoluto (MAE)
    mae = np.mean(np.abs(altitudes - setpoint))
    # Tempo de Estabilização (Settling Time)
    tolerance = 0.05 * setpoint  # 5% do setpoint
    settling_time_index = np.where(np.abs(altitudes - setpoint) > tolerance)[0]
    if len(settling_time_index) == 0:
        settling_time = tempo[-1]
    else:
        settling_time = tempo[settling_time_index[-1]]
    # Overshoot Máximo
    #overshoot = np.max(altitudes) - setpoint
    overshoot = np.min(altitudes) - setpoint #para altitudes maiores que 10 m
    # Erro Quadrático Médio (RMSE)
    rmse = np.sqrt(np.mean((altitudes - setpoint) ** 2))
    return mae, settling_time, overshoot, rmse


# Configuração do sistema e do PID
pid = PID(kp=5, ki=1.2, kd=0.8)
drone = DroneAltitude(m=1.5, Cd=0.1)
# Parâmetros
dt = 0.05  # Passo de tempo
tempo = np.arange(0, 50, dt)  # Tempo total de simulação

# Simulação
setpoint = 10  # Altura desejada (em metros)
altitudes_pid = []
for t in tempo:
    control_signal = pid.compute(setpoint, drone.altitude, dt)
    altitude_pid = drone.update(control_signal, dt)
    altitudes_pid.append(altitude_pid)


# Resultados gráficos PID
plt.plot(tempo, altitudes_pid, label="Altura")
plt.axhline(y=setpoint, color='r', linestyle='--', label="Setpoint")
plt.xlabel("Tempo (s)")
plt.ylabel("Altura (m)")
plt.title("Controle PID para Altura de Drone")
plt.legend()
plt.grid()
plt.show()
# Exibição das métricas PID
metrics_pid = calculate_metrics(tempo, altitudes_pid, setpoint)
print("\nMétricas de Desempenho:\n")
print("Controlador PID:")
print(f"Erro Médio Absoluto (MAE): {metrics_pid[0]:.3f}")
print(f"Tempo de Estabilização: {metrics_pid[1]:.3f} s")
print(f"Overshoot Máximo: {metrics_pid[2]:.3f} m")
print(f"Erro Quadrático Médio (RMSE): {metrics_pid[3]:.3f}")


class DroneAltitudefuzzy:
    def __init__(self, m, Cd):
        self.altitude = 0 
        self.velocidade = 0
        self.m = m
        self.Cd = Cd
    def update2(self, impulso, dt):
        g = 9.81
        impulso = np.clip(impulso, 0, 100)
        aceleracao = (impulso - self.Cd * self.velocidade - self.m * g) / self.m
        self.velocidade += aceleracao * dt
        self.altitude += self.velocidade * dt
        return self.altitude, aceleracao
    
 # Configuração do sistema   
drone = DroneAltitudefuzzy(m=1.5, Cd=0.1)
# Variáveis Fuzzy
altitude = ctrl.Antecedent(np.arange(0, 21, 1), 'altitude')
aceleracao = ctrl.Antecedent(np.arange(-10, 11, 1), 'aceleracao')
impulso = ctrl.Consequent(np.arange(0, 101, 1), 'impulso')

# Funções de pertinência para Altitude
altitude['muito_baixa'] = fuzz.trapmf(altitude.universe, [0, 0, 2, 5])
altitude['baixa'] = fuzz.trapmf(altitude.universe, [2, 7, 9, 10])
altitude['ideal'] = fuzz.trimf(altitude.universe, [9, 10, 11])
altitude['alta'] = fuzz.trapmf(altitude.universe, [10, 12, 14, 16])
altitude['muito_alta'] = fuzz.trapmf(altitude.universe, [15, 18, 20, 20])

# Funções de pertinência para Aceleração
aceleracao['queda_grande'] = fuzz.trapmf(aceleracao.universe, [-10, -10, -7, -3])
aceleracao['queda_pequena'] = fuzz.trapmf(aceleracao.universe, [-5, -3.2, -2.8, 0])
aceleracao['zero'] = fuzz.trimf(aceleracao.universe, [-2, 0, 2])
aceleracao['subida_pequena'] = fuzz.trapmf(aceleracao.universe, [0, 2.8, 3.2, 5])
aceleracao['subida_grande'] = fuzz.trapmf(aceleracao.universe, [3, 7, 10, 10])

# Funções de pertinência para impulso (Empuxo)
impulso['fraco'] = fuzz.trapmf(impulso.universe, [0, 0, 5, 12])
impulso['medio'] = fuzz.trimf(impulso.universe, [8, 15, 22])
impulso['forte'] = fuzz.trapmf(impulso.universe, [20, 40, 100, 100])

altitude.view()
aceleracao.view()
impulso.view()

# Regras Fuzzy
regra1 = ctrl.Rule(altitude['muito_baixa'] & aceleracao['queda_grande'], impulso['forte'])
regra2 = ctrl.Rule(altitude['muito_baixa'] & aceleracao['queda_pequena'], impulso['forte'])
regra3 = ctrl.Rule(altitude['muito_baixa'] & aceleracao['zero'], impulso['forte'])
regra4 = ctrl.Rule(altitude['muito_baixa'] & aceleracao['subida_pequena'], impulso['forte'])
regra5 = ctrl.Rule(altitude['muito_baixa'] & aceleracao['subida_grande'], impulso['medio'])

regra6 = ctrl.Rule(altitude['baixa'] & aceleracao['queda_grande'], impulso['forte'])
regra7 = ctrl.Rule(altitude['baixa'] & aceleracao['queda_pequena'], impulso['medio'])
regra8 = ctrl.Rule(altitude['baixa'] & aceleracao['zero'], impulso['medio'])
regra9 = ctrl.Rule(altitude['baixa'] & aceleracao['subida_pequena'], impulso['medio'])
regra10 = ctrl.Rule(altitude['baixa'] & aceleracao['subida_grande'], impulso['medio'])

regra11 = ctrl.Rule(altitude['ideal'] & aceleracao['queda_grande'], impulso['forte'])
regra12 = ctrl.Rule(altitude['ideal'] & aceleracao['queda_pequena'], impulso['medio'])
regra13 = ctrl.Rule(altitude['ideal'] & aceleracao['zero'], impulso['medio'])
regra14 = ctrl.Rule(altitude['ideal'] & aceleracao['subida_pequena'], impulso['medio'])
regra15 = ctrl.Rule(altitude['ideal'] & aceleracao['subida_grande'], impulso['fraco'])

regra16 = ctrl.Rule(altitude['alta'] & aceleracao['queda_grande'], impulso['medio'])
regra17 = ctrl.Rule(altitude['alta'] & aceleracao['queda_pequena'], impulso['medio'])
regra18 = ctrl.Rule(altitude['alta'] & aceleracao['zero'], impulso['fraco'])
regra19 = ctrl.Rule(altitude['alta'] & aceleracao['subida_pequena'], impulso['fraco'])
regra20 = ctrl.Rule(altitude['alta'] & aceleracao['subida_grande'], impulso['fraco'])

regra21 = ctrl.Rule(altitude['muito_alta'] & aceleracao['queda_grande'], impulso['fraco'])
regra22 = ctrl.Rule(altitude['muito_alta'] & aceleracao['queda_pequena'], impulso['fraco'])
regra23 = ctrl.Rule(altitude['muito_alta'] & aceleracao['zero'], impulso['fraco'])
regra24 = ctrl.Rule(altitude['muito_alta'] & aceleracao['subida_pequena'], impulso['fraco'])
regra25 = ctrl.Rule(altitude['muito_alta'] & aceleracao['subida_grande'], impulso['fraco'])

# Controle Fuzzy
impulso_control = ctrl.ControlSystem([regra1, regra2, regra3, regra4, regra5,
                                     regra6, regra7, regra8, regra9, regra10,
                                     regra11, regra12, regra13, regra14, regra15,
                                     regra16, regra17, regra18, regra19, regra20,
                                     regra21, regra22, regra23, regra24, regra25])
impulso_sim = ctrl.ControlSystemSimulation(impulso_control)

impulso_sim.input['altitude'] = 0
impulso_sim.input['aceleracao'] = 0

# Calcula a saída do sistema de controle fuzzy
impulso_sim.compute()
impulso.view(sim=impulso_sim)
plt.show()

# Parâmetros de simulação
dt = 0.05  # Passo de tempo
tempo = np.arange(0, 50, dt)  # Tempo total de simulação
setpoint = 10  # Altura desejada (em metros)
altitudes_fuzzy = []
aceleraçoes = []

# Simulação
for t in tempo:
    error = setpoint - drone.altitude
    impulso_sim.input['altitude'] = drone.altitude
    impulso_sim.input['aceleracao'] = drone.velocidade / dt
    impulso_sim.compute()
    impulso = impulso_sim.output['impulso']
    altitude, aceleracao = drone.update2(impulso, dt)
    altitudes_fuzzy.append(altitude)
    aceleraçoes.append(aceleracao)


# Resultados
plt.figure(figsize=(10, 6))
plt.plot(tempo, altitudes_fuzzy, label="Altura")
plt.axhline(y=setpoint, color='r', linestyle='--', label="Setpoint")
plt.xlabel("Tempo (s)")
plt.ylabel("Altura (m)")
plt.title("Controle Fuzzy para Altura de Drone")
plt.legend()
plt.grid()
plt.show()
metrics_fuzzy = calculate_metrics(tempo, altitudes_fuzzy, setpoint)
# Exibição dos resultados
print("\nControlador Fuzzy:")
print(f"Erro Médio Absoluto (MAE): {metrics_fuzzy[0]:.3f}")
print(f"Tempo de Estabilização: {metrics_fuzzy[1]:.3f} s")
print(f"Overshoot Máximo: {metrics_fuzzy[2]:.3f} m")
print(f"Erro Quadrático Médio (RMSE): {metrics_fuzzy[3]:.3f}")


# Visualização das altitudes
plt.figure(figsize=(10, 6))
plt.plot(tempo, altitudes_pid, label="PID")
plt.plot(tempo, altitudes_fuzzy, label="Fuzzy", linestyle='--')
plt.axhline(y=setpoint, color='r', linestyle='--', label="Setpoint")
plt.xlabel("Tempo (s)")
plt.ylabel("Altura (m)")
plt.title("Comparação de Controladores PID e Fuzzy")
plt.legend()
plt.grid()
plt.show()
