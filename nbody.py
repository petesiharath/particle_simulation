import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Particle:
    """
    Class for a particle.
    
    Attributes:
        pos (numpy array): 
            3D position of the particle.

        vel (numpy array):
            3D velocity of the particle.
        
        stored_x (list):
            List of stored x values. Used for plotting the trajectory of the particle.
    """

    def __init__(self, pos, vel):
        """
        Initialise a particle with a position and velocity .
        
        Args:
            pos (numpy array): 
                3D position of the particle.

            vel (numpy array):
                3D velocity of the particle.
        """

        self.pos = pos
        self.vel = vel


class System:
    """
    Class for a simulating a system of particles.

    Attributes:
        particles (list):
            List of particles in the system.

        num_particles (int):
            Number of particles in the system.

        times (list):
            List of times at which the system is evaluated.

        vel (numpy array):
            3D array of velocities of the particles.

        pos (numpy array):
            3D array of positions of the particles.

        acc (numpy array):
            3D array of accelerations of the particles.
        
        stored_pos (numpy array):
            3D array of stored positions of the particles.
        
        stored_vel (numpy array):
            3D array of stored velocities of the particles.
        
        stored_flow_vel (numpy array):
            1D array of stored flow velocities of the particles.
        
        num_steps (int):
            Number of steps the simulation is run for.
        
        dt (float):
            Timestep of the simulation.
        
        box_length (float):
            Length of the box in which the particles are contained.
        
        avg_temperature (float):
            Average temperature of the system.
        
        avg_pressure (float):
            Average pressure of the system.
    
    Methods:
        validate_spawn:
            Validate the spawn of a particle.
        
        spawn_particles:
            Spawn a given number of particles.
        
        force_coefficient:
            Calculate the force coefficient for a given distance.

        calculate_force:
            Calculate the force on a particle due to another particle.

        calculate_acceleration:
            Calculate the acceleration of a particle due to all other particles.
        
        calculate_temperature:
            Calculate the average temperature of the system over the whole simulation.
        
        calculate_pressure:
            Calculate the average pressure of the system over the whole simulation.

        randomise_velocities:
            Randomise the velocities of the particles.

        run:
            Run the simulation for a given number of timesteps.

        plot:
            Plot the trajectory of the particles.
        
        measure:
            Measure the average temperature and pressure of the system.
    """

    def __init__(self, particles, box_length, num_steps, dt):
        """
        Initialise a system with a list of particles, box length, number of steps and timestep.

        Args:
            particles (list):
                List of particles in the system.

            box_length (float):
                Length of the box in which the particles are contained.

            num_steps (int):
                Number of steps the simulation is run for.

            dt (float):
                Timestep of the simulation.
        """

        self.particles = particles
        self.num_particles = len(particles)

        self.dt = dt
        self.num_steps = num_steps

        self.box_length = box_length

        # Initialising arrays to store the positions, velocities and flow velocities of the particles
        self.stored_pos = np.zeros((num_steps, self.num_particles, 3))
        self.stored_vel = np.zeros((num_steps, self.num_particles, 3))
        self.stored_flow_vel = np.zeros((num_steps, self.num_particles, 3))
        
        self.times = []

        self.vel = np.zeros((self.num_particles, 3))
        self.pos = np.zeros((self.num_particles, 3))
        self.acc = np.zeros((self.num_particles, 3))

        # Storing the initial positions and velocities of the particles
        for i in range(self.num_particles):
            self.pos[i] = particles[i].pos
            self.vel[i] = particles[i].vel
    

    def validate_spawn(self, particle):
        """
        Validate the spawn of a particle.
        
        Args:
            particle (Particle):
                Particle to be spawned.
        
        Returns:
            bool:
                True if the particle can be spawned, False otherwise.
        """
            
        for i in range(self.num_particles):
            # Check if the particle is too close to another particle
            if np.linalg.norm(particle.pos - self.particles[i].pos) < 1:
                return False
        return True
    

    def spawn_particles(self, n):
        """
        Spawn a given number of particles.

        Args:
            n (int):
                Number of particles to spawn.
        """

        for i in range(n):
            valid = False
            # Keep trying to spawn a particle within box until it is valid
            while not valid:
                pos = np.random.uniform(-self.box_length / 2, self.box_length / 2, 3)
                vel = np.zeros(3)
                particle = Particle(pos, vel)
                valid = self.validate_spawn(particle)

            self.particles.append(particle)
            self.num_particles += 1
        
        # Reinitialise arrays to store the positions, velocities and flow velocities of the particles
        self.stored_pos = np.zeros((self.num_steps, self.num_particles, 3))
        self.stored_vel = np.zeros((self.num_steps, self.num_particles, 3))
        self.stored_flow_vel = np.zeros(self.num_steps)
        
        # Reinitialise arrays to store the positions, velocities and accelerations of the particles
        self.pos = np.zeros((self.num_particles, 3))
        self.vel = np.zeros((self.num_particles, 3))
        self.acc = np.zeros((self.num_particles, 3))

        # Storing the initial positions and velocities of the particles
        for i in range(self.num_particles):
            self.pos[i] = self.particles[i].pos
            self.vel[i] = self.particles[i].vel


    def force_coefficient(self, r):
        """
        Calculate the force coefficient for a given distance.

        Args:
            r (float):
                Distance between two particles.
        
        Returns:
            float:
                Force coefficient for the given distance.
        """

        return 24 * (-2 * (1 / r) ** 13 + (1 / r) ** 7)


    def calculate_force(self, pos_p, pos_j):
        """
        Calculate the force on a particle due to another particle.

        Args:
            pos_p (numpy array):
                Position of the particle on which the force is acting.

            pos_j (numpy array):
                Position of the particle causing the force.
        
        Returns:
            numpy array:
                Force acting on the particle.
        """

        r = np.linalg.norm(pos_j - pos_p)
        r_hat = (pos_j - pos_p) / r
        return self.force_coefficient(r) * r_hat


    def calculate_acceleration(self, pos_p, p):
        """
        Calculate the acceleration of a particle due to all other particles.

        Args:
            pos_p (numpy array):
                Position of the particle.

            p (int):
                Index of the particle.
        """

        self.acc[p] = np.zeros(3)
        for j in range(self.num_particles):
            if p != j:
                self.acc[p] += self.calculate_force(pos_p, self.pos[j])

    
    def calculate_temperature(self):
        """
        Calculate the average temperature of the system over the whole simulation.
        """

        sum_v2 = np.sum(self.stored_vel * self.stored_vel)
        self.avg_temperature = sum_v2 / (3 * self.num_particles * self.num_steps)


    def calculate_pressure(self):
        """
        Calculate the average pressure of the system over the whole simulation.
        """

        avg_momentum = np.sum(self.stored_flow_vel) / self.num_steps
        self.avg_pressure = 3 * avg_momentum / (self.box_length ** 2 * self.dt)


    def randomise_velocities(self, vel_mean):
        """
        Randomise the velocities of the particles.

        Args:
            vel_mean (numpy array):
                Mean 3D velocity of the particles.
        """

        # Randomly initialise velocities of the particles normally distributed around the mean velocity
        self.vel = np.random.normal(vel_mean, 1, (self.num_particles, 3))

        # Randomly flip the sign of the velocities
        random_negatives = np.random.choice([1, -1], (self.num_particles, 3))
        self.vel = self.vel * random_negatives


    def run(self):
        """
        Run the simulation for a given number of timesteps.
        """

        for t in range(self.num_steps):
            # Store the time at which the system is evaluated
            self.times.append(t * self.dt)

            # Calculate the acceleration of each particle
            for p in range(self.num_particles):
                self.calculate_acceleration(self.pos[p], p)

            # Update the position and velocity of each particle
            self.vel += self.acc * self.dt * 0.5
            self.pos += self.vel * self.dt

            # Recalculate the acceleration of each particle
            for p in range(self.num_particles):
                self.calculate_acceleration(self.pos[p], p)

            # Update the velocity of each particle
            self.vel += self.acc * self.dt * 0.5

            # Reflect the particles off the walls of the box
            self.vel[np.abs(self.pos) >= self.box_length / 2] *= -1
            self.pos[self.pos > self.box_length / 2] = self.box_length / 2
            self.pos[self.pos < -self.box_length / 2] = -self.box_length / 2

            # Store the positions, velocities of the particles
            self.stored_pos[t] = self.pos
            self.stored_vel[t] = self.vel
            
            # Store the flow velocity of the particles
            condition_1 = self.pos[:, 0] <= 0 
            condition_2 = self.pos[:, 0] + self.vel[:, 0] * self.dt > 0
            condition = condition_1 & condition_2
            self.stored_flow_vel[t] = np.sum(self.vel[condition, 0])


    def plot(self):
        """
        Plot the trajectory of the particles.
        """
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Plot the trajectory of each particle
        for i in range(self.num_particles):
            x = self.stored_pos[:, i, 0]
            y = self.stored_pos[:, i, 1]
            z = self.stored_pos[:, i, 2]
            ax.plot(x, y, z, label=f"Particle {i + 1}")

        # Plot the box
        box_corners = np.array([[-self.box_length/2, -self.box_length/2, -self.box_length/2],
                        [-self.box_length/2, -self.box_length/2, self.box_length/2],
                        [-self.box_length/2, self.box_length/2, -self.box_length/2],
                        [-self.box_length/2, self.box_length/2, self.box_length/2],
                        [self.box_length/2, -self.box_length/2, -self.box_length/2],
                        [self.box_length/2, -self.box_length/2, self.box_length/2],
                        [self.box_length/2, self.box_length/2, -self.box_length/2],
                        [self.box_length/2, self.box_length/2, self.box_length/2]])

        box_edges = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
        for edge in box_edges:
            ax.plot3D(box_corners[edge, 0], box_corners[edge, 1], box_corners[edge, 2], 'k--', alpha=0.5)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-self.box_length / 2, self.box_length / 2)
        ax.set_ylim(-self.box_length / 2, self.box_length / 2)
        ax.set_zlim(-self.box_length / 2, self.box_length / 2)
        ax.legend()


    def plot_2d(self):
        """
        Plot the x trajectory of the particles.
        """

        fig, ax = plt.subplots()

        # Plot the x trajectory of each particle
        for i in range(self.num_particles):
            ax.plot(self.times, self.stored_pos[:, i, 0], label=f"Particle {i + 1}")
            ax.set_xlabel('Time')
            ax.set_ylabel('X position')

    
    def measure(self, velocity):
        """
        Measure the average temperature and pressure of the system.

        Args:
            velocity (float):
                Mean 3D velocity of the particles.
        
        Returns:
            float:
                Average temperature of the system.
            
            float:
                Average pressure of the system.
        """

        self.randomise_velocities(velocity)
        self.run()
        self.calculate_temperature()
        self.calculate_pressure()

        return self.avg_temperature, self.avg_pressure