import taichi as ti

@ti.data_oriented
class FlowFieldSimulator:
    def __init__(self, num_particles: int = 50000):
        self.num_particles = num_particles
        self.dt = 0.0035

        # Partikel-Zustände auf GPU
        self.pos = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
        self.drag = ti.field(dtype=ti.f32, shape=num_particles)  # Lokaler Strömungswiderstand
        self.life = ti.field(dtype=ti.f32, shape=num_particles)
        self.color = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)

        # Maus-Interaktion
        self.mouse_pos = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.mouse_active = ti.field(dtype=ti.i32, shape=())
        self.time = ti.field(dtype=ti.f32, shape=())

    @ti.kernel
    def init(self):
        self.time[None] = 0.0
        self.mouse_active[None] = 0
        self.mouse_pos[None] = ti.Vector([0.5, 0.5])
        for i in range(self.num_particles):
            self.reset_particle(i, initial=True)

    @ti.func
    def reset_particle(self, i: int, initial: bool):
        if initial:
            self.pos[i] = ti.Vector([ti.random(ti.f32) * 1.05 - 0.02, ti.random(ti.f32)])
        else:
            self.pos[i] = ti.Vector([-0.02, ti.random(ti.f32)])
        self.vel[i] = ti.Vector([0.9, 0.0])
        self.drag[i] = 0.0
        self.life[i] = ti.random(ti.f32) * 1.0

    @ti.func
    def sample_vector_field(self, p: ti.template(), t: ti.f32):
        scale = 2.4
        a1 = ti.sin(p[0] * scale + t * 0.35) * ti.cos(p[1] * scale - t * 0.25)
        a2 = ti.sin(p[1] * scale * 1.8 + t * 0.2) * 0.4
        angle = (a1 + a2) * 1.8
        
        base_v = ti.Vector([ti.cos(angle) + 1.15, ti.sin(angle) * 0.65])
        local_drag = 0.08  # Grund-Scherung im Freistrom

        if self.mouse_active[None] == 1:
            m_vec = p - self.mouse_pos[None]
            dist = m_vec.norm()
            r_influence = 0.26
            
            if dist < r_influence:
                norm_d = dist / r_influence
                smooth_factor = (1.0 - norm_d * norm_d) ** 2

                inv_d = 1.0 / (dist + 1e-3)
                swirl = ti.Vector([-m_vec[1], m_vec[0]]) * inv_d
                push = m_vec * inv_d
                
                base_v = base_v * (1.0 - 0.7 * smooth_factor) + (swirl * 2.4 + push * 0.9) * smooth_factor
                
                # Hohe Dissipation und Verwirbelung = massiver lokaler Widerstand
                local_drag += smooth_factor * 1.25

        return base_v, local_drag

    @ti.func
    def colormap_aerodynamic(self, val: ti.f32) -> ti.template():
        # Kontinuierliche Farbskala: Grün (0.0) -> Gelb (0.45) -> Orange (0.7) -> Rot (1.0)
        c = ti.Vector([0.0, 0.0, 0.0])
        clamped = ti.max(0.0, ti.min(1.0, val))
        
        if clamped < 0.45:
            # Grün nach Gelb
            t = clamped / 0.45
            c = ti.Vector([0.15 + 0.80 * t, 0.88 + 0.08 * t, 0.20 * (1.0 - t)])
        elif clamped < 0.75:
            # Gelb nach Orange
            t = (clamped - 0.45) / 0.30
            c = ti.Vector([0.95 + 0.05 * t, 0.92 - 0.42 * t, 0.05])
        else:
            # Orange nach Tiefrot
            t = (clamped - 0.75) / 0.25
            c = ti.Vector([1.0, 0.50 - 0.35 * t, 0.05 + 0.15 * t])
            
        return c

    @ti.kernel
    def step(self):
        t = self.time[None]
        for i in range(self.num_particles):
            target_v, drag_val = self.sample_vector_field(self.pos[i], t)
            
            # Beschleunigungs-Gradient fließt in den Widerstand ein
            dv = (target_v - self.vel[i]).norm()
            total_resistance = drag_val + dv * 0.45
            self.drag[i] = self.drag[i] * 0.90 + total_resistance * 0.10

            self.vel[i] = self.vel[i] * 0.94 + target_v * 0.06
            self.pos[i] += self.vel[i] * self.dt
            self.life[i] += self.dt * 0.25

            if self.pos[i][0] > 1.03 or self.pos[i][1] < -0.03 or self.pos[i][1] > 1.03 or self.life[i] > 1.0:
                self.reset_particle(i, initial=False)

            # Farb-Lookup: Grün (optimal) bis Rot (hoher Widerstand)
            metric = ti.min(self.drag[i] / 0.85, 1.0)
            self.color[i] = self.colormap_aerodynamic(metric)

        self.time[None] += self.dt * 1.0