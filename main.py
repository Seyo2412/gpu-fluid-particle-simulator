import taichi as ti
from src.flow_field import FlowFieldSimulator

ti.init(arch=ti.metal)

sim = FlowFieldSimulator(num_particles=50000)
sim.init()

window = ti.ui.Window("Aerodynamic Flow & Drag Field Analyzer", (950, 900), vsync=True)
gui = window.get_gui()
canvas = window.get_canvas()

print("CFD Strömungssimulation aktiv.")
print("Farbskala: GRÜN = Laminar / Geringer Widerstand -> ROT = Turbulenz / Hoher Staudruck")

while window.running:
    if window.is_pressed(ti.ui.LMB):
        cursor = window.get_cursor_pos()
        sim.mouse_pos[None] = [cursor[0], cursor[1]]
        sim.mouse_active[None] = 1
    else:
        sim.mouse_active[None] = 0

    for _ in range(2):
        sim.step()

    # Moderner Carbon-/Dark-Studio Hintergrund
    canvas.set_background_color((0.03, 0.03, 0.04))

    # Partikel rendern
    canvas.circles(
        sim.pos,
        radius=0.0016,
        per_vertex_color=sim.color
    )

    # Clean UI Overlay / Telemetrie-Skala
    with gui.sub_window("Aero Drag Index", 0.03, 0.03, 0.28, 0.16) as sub:
        gui.text("Airflow Resistance: q = 0.5 * rho * v²")
        gui.text("Low Drag (Laminar) : GREEN")
        gui.text("Transition         : YELLOW")
        gui.text("High Drag (Vortex) : RED")

    window.show()