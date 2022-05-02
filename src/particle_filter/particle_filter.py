import trimesh
import numpy as np
import matplotlib.pyplot as plt
import random
class ParticleFilter:
    def __init__(self, num_particles, initial_guess, sample_std, mesh_fn = lambda pose: trimesh.primitives.Cylinder(radius=0.0762/2, height=0.12065, transform=pose)) -> None:
        self.num_particles = num_particles
        self.initial_guess = initial_guess
        self.mesh_fn = mesh_fn
        self.sample_std = sample_std
        # create initial particle
        self.particles = [mesh_fn(self.initial_guess)]
        self.weights = [1]
        # sample initial set of particles from initial guess
        self.particles = self.create_particles()
    
    def create_particles(self):
        particles = []
        for _ in range(self.num_particles):
            base_particle = np.random.choice(self.particles, p = self.weights)
            sampled_particle = self.sample_particle(base_particle)
            particles.append(sampled_particle)
        return particles

    def sample_particle(self, base_particle):
        # generate a uniformly random rotation
        rand_rot_perturb, _ = np.linalg.qr(np.random.randn([3, 3]), mode='complete')
        rand_pos_perturb = np.random.normal(scale=self.sample_std, size=[3])
        perturbation = np.random.zeros([4, 4])
        perturbation[:3, :3] = rand_rot_perturb
        perturbation[:3, -1] = rand_pos_perturb
        perturbation[3, 3] = 1
        new_particle = base_particle.copy().apply_transform(perturbation)
        return new_particle

    def reweight_particle(self, particle_idx, mult_factor):
        self.weights[particle_idx] *= mult_factor
        total_weight = sum(self.weights)
        #renormalize weights
        self.weights = [p / total_weight for p in self.weights]
            
