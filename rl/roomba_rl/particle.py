from typing import NamedTuple
import random
import math

class Pose(NamedTuple):
    x: float
    y: float
    theta: float

class RandomParticle():
    def __init__(self, pos: Pose, speed: float):
        self.pose = Pose(x=pos.x, y=pos.y, theta=pos.theta)
        self._rnd = random.Random(int(pos.x) + 1024*int(pos.y))
        self._speed = speed # Speed per step

    def move(self):
        dist = self._speed 
        theta = self._rnd.random() * 2 * math.pi
        prev_pos = self.pose
        self.pose = Pose(
            x=prev_pos.x + dist * math.cos(theta),
            y=prev_pos.y + dist * math.sin(theta),
            theta=theta
        )
        return self.pose

class ParticleMap():
    def __init__(self, n_particles, x_bound, y_bound, max_dist=2, collision_dist=4):
        self._particles = []
        for i in range(n_particles):
            start_pos = Pose(
                x=random.random() * x_bound, 
                y=random.random() * y_bound,
                theta=0
            )   
            self._particles.append(
                RandomParticle(start_pos, max_dist*random.random())
            )
        self._collision_dist = collision_dist

    def move(self):
        for p in self._particles: 
            p.move()

    def detect_collision(self, pose):
        for particle in self._particles:
            dist = (particle.pose.x - pose.x)**2 + (particle.pose.y - pose.y)**2
            if dist < self._collision_dist ** 2:
                return True
        return False

    def get_readings(self, pose):
        readings = []
        for particle in self._particles:
            p = particle.pose
            angle = math.atan2(p.y - pose.y, p.x - pose.x)
            dist = (p.x - pose.x)**2 + (p.y - pose.y)**2
            readings.append((angle, math.sqrt(dist)))
        return readings

    @property
    def particles(self):
        return [p.pose for p in self._particles]

if __name__ == "__main__":
    p = RandomParticle(Pose(x=1, y=2, theta=0), speed=1)
    orig_pos = p.pos
    p.move() 
    print(p.pos)
    print((p.pos.x-orig_pos.x)**2 + (p.pos.y-orig_pos.y)**2)
    
    pmap = ParticleMap(5, 5, 10)
    pmap.move()
    print([r.pos for r in pmap._particles])
