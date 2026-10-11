import taichi as ti
import numpy as np
from src.flow_field import FlowFieldSimulator

def test_initialization():
    ti.init(arch=ti.cpu)
    sim = FlowFieldSimulator(num_particles=1000)
    sim.init()
    
    positions = sim.pos.to_numpy()
    assert positions.shape == (1000, 2)
    assert np.all(positions[:, 1] >= 0.0)
    assert np.all(positions[:, 1] <= 1.0)

def test_simulation_step_stability():
    sim = FlowFieldSimulator(num_particles=500)
    sim.init()
    
    # 10 Zeitschritte durchfuehren
    for _ in range(10):
        sim.step()
        
    positions = sim.pos.to_numpy()
    velocities = sim.vel.to_numpy()
    
    # Pruefen, dass keine NaN-Werte oder Endlos-Werte entstanden sind
    assert not np.isnan(positions).any()
    assert not np.isinf(positions).any()
    assert not np.isnan(velocities).any()
    assert not np.isinf(velocities).any()