import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parent.parent))

import taichi as ti
from src.flow_field import FlowFieldSimulator

os.makedirs("assets", exist_ok=True)

# Headless / Offscreen Initialisierung
ti.init(arch=ti.metal)

num_particles = 40000
sim = FlowFieldSimulator(num_particles=num_particles)
sim.init()

frames = 75
width, height = 700, 700
rendered_images = []

print(f"Rendere {frames} Frames fuer das Portfolio-GIF (40k Partikel)...")

for frame in range(frames):
    # Virtuelle Mausbewegung im Kreis (erzeugt dynamische Wirbel)
    angle = frame * (2.0 * np.pi / frames)
    m_x = 0.5 + 0.18 * np.cos(angle)
    m_y = 0.5 + 0.14 * np.sin(angle)
    
    sim.mouse_pos[None] = [float(m_x), float(m_y)]
    sim.mouse_active[None] = 1

    # 2 Substeps berechnen
    for _ in range(2):
        sim.step()

    # Partikeldaten von GPU nach CPU holen
    positions = sim.pos.to_numpy()
    colors = (sim.color.to_numpy() * 255.0).astype(np.uint8)

    # Bild erzeugen (Dark Canvas)
    img = Image.new("RGB", (width, height), (8, 8, 12))
    draw = ImageDraw.Draw(img)

    # Skalierung & Zeichnen
    px = np.clip((positions[:, 0] * width).astype(int), 0, width - 1)
    py = np.clip(((1.0 - positions[:, 1]) * height).astype(int), 0, height - 1)

    # Schnelles Batch-Zeichnen mit feinen Punkten
    for i in range(0, num_particles, 2):
        x, y = px[i], py[i]
        c = tuple(colors[i])
        draw.rectangle([x, y, x + 1, y + 1], fill=c)

    rendered_images.append(img)
    print(f"Frame {frame + 1}/{frames} gerendert", end="\r")

print("\nSpeichere assets/demo_flow.gif ...")
gif_path = "assets/demo_flow.gif"
rendered_images[0].save(
    gif_path,
    save_all=True,
    append_images=rendered_images[1:],
    duration=35,
    loop=0,
    optimize=True
)
print(f"Erfolgreich erstellt: {gif_path}")