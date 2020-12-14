import numpy as np
import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D

class NbodyClass:
    def __init__(self, x, v, m=1, sgrid=10, dt=0.1, outdir='outputs/', periodic=False, guard=2):
        self.x = x.copy() #position vector.have x[0, :] = x, x[1,:] = y, x[2, :]=z etc.
        self.x_half = x.copy()
        self.v = v.copy() #velocity vector. 
        self.v_half = v.copy()
        self.m = m        #mass of a single particle
        self.dt = dt      #time step
        self.steps_taken = 0
        
        #defining the grid
        #ONLY WANT GUARD CELLS FOR NON PERIODIC
        if periodic:
            self.sgrid = sgrid #the smaller grid of actual data
            self.ngrid = sgrid #the grid with gaurd cells
        else:
            self.sgrid = sgrid #the smaller grid of actual data
            self.ngrid = sgrid + guard #the grid with gaurd cells
            
        self.grid = np.zeros((self.ngrid, self.ngrid, self.ngrid))
        self.update_grid()
        
        
        #deal with periodicity
        self.periodic = periodic
        
        #pre-define acceleration
        self.acc = np.zeros(x.shape)
        
        #where to save output images
        self.outdir = outdir
        
        #calculate greens of the laplacian for a single particle
        #creates self.greens_pot
        self.greens_pot = 0
        self.calculate_greens()
        
        #initialize potential
        self.pot = np.zeros((self.ngrid, self.ngrid, self.ngrid))
                
    def calculate_greens(self):
        if self.periodic:
            num = self.ngrid
        else: #NOT periodic. will need to pad extra values of green's function
            num = 2*self.ngrid-1
            
        dx = np.arange(num)            
        #making it wrap around. stuff in same column/row/depth will have 0
        #previous wraps around to -1, next goes to 1
        dx[num//2:] = dx[num//2:]-num
            
        #distances away from particle in x, y, z
        #particle lives at 0, 0, 0
        xmesh, ymesh, zmesh = np.meshgrid(dx,dx,dx)
        dr = np.sqrt(xmesh**2 + ymesh**2 + zmesh**2)

        #point at 0,0,0 is 0 and will blow up
        #this is essentially softening potential, when it's that grid cell it's 1
        dr[0, 0, 0] = 1
        #define potential to be negative and go to zero infinitely far away
        self.greens_pot = -1/(4*np.pi*dr)
    
    #convolve the density matrix with the potential
    def get_potential(self, DEBUG=False):
        #FIRST UPDATE: self.grid (density matrix)
        #convolution in time domain equal to multiplication in frequency domain
        
        #probably need to know periodic or nonperiodic
        #nonperiodic is going to need to be zero padded
        
        if self.periodic: #periodic doesn't need padding
            fft_greens = np.fft.fftn(self.greens_pot)
            fft_density = np.fft.fftn(self.grid)

            self.pot = -1*np.abs(np.fft.ifftn(fft_density * fft_greens))
        else: #not periodic, need padding
            N = 2*self.ngrid - 1 #everything is square
    
            # DO NOT PAD GREENS POTENTIAL WITH ZEROS
            #WHY? because it's NOT zero off the grid. dies off slowly.
            #Instead in creation it has already been calculated to size N.

            #padding density: pad with zeros, assume particles are gone off grid
            #TO DO: keep BUFFER CELLS around the edge and pad with THOSE, and THEN zeros.
            density_padded = np.zeros( (N, N, N) )
            density_padded[0:self.ngrid, 0:self.ngrid, 0:self.ngrid] = self.grid
            
            fft_greens = np.fft.fftn(self.greens_pot)
            fft_density = np.fft.fftn(density_padded)

            pot_padded = np.fft.ifftn(fft_density * fft_greens)
            self.pot = -1*np.abs(pot_padded[0:self.ngrid, 0:self.ngrid, 0:self.ngrid])
            
            if DEBUG:
                mpl.imshow(-1*np.abs(pot_padded[0, :, :]), cmap='hot')
                mpl.title('PADDED final potential x=0 2D slice')
                mpl.show()

        if DEBUG:
            mpl.imshow(self.pot[0, :, :], cmap='hot')
            mpl.title('final potential x=0 2D slice')
            mpl.show()

            mpl.imshow(self.pot[self.ngrid//2, :, :], cmap='hot')
            mpl.title('final potential x=' + repr(self.ngrid//2) + ' 2D slice' )
            mpl.show()

            print('the particles are at ', np.nonzero(self.grid))        
        return 0
    
    def calculate_acceleration(self, DEBUG=False):
        #UPDATE THE GRADIENT FIRST. (and for the gradient, the density grid)
        #take the GRADIENT of the potential to get the force. divide by mass
        #use the half step of x to do the calculation
        num = self.x_half.shape[1]
        
        for ii in range(0, num):
            x_idx = tuple(self.x_half[:, ii].astype(int))
            part_pot = np.roll(self.greens_pot, x_idx, (0, 1, 2))
            potential = self.pot - part_pot
            
            if DEBUG:
                mpl.figure()
                mpl.imshow(self.greens_pot[0, :, :], cmap='hot')
                mpl.title('greens pot, x=0')
                mpl.colorbar()
                mpl.show()

                mpl.figure()
                mpl.imshow(part_pot[0, :, :], cmap='hot')
                mpl.title('part_pot, x=0' + repr( self.x_half[:, ii].astype(int)) )
                mpl.colorbar()
                mpl.show()

                mpl.figure()
                mpl.imshow(self.pot[0, :, :], cmap='hot')
                mpl.title('current potential of whole system, x=0')
                mpl.colorbar()
                mpl.show()

                mpl.figure()
                mpl.imshow(potential[0, :, :], cmap='hot')
                mpl.title('part_total, total potential particle sees, x=0')
                mpl.colorbar()
                mpl.show()
                    
            #GRADIENT: TWO possible methods.
            #1: np.gradient. cons: calculates for ENTIRE matrix. but our matrix
            #is DIFFERENT every time because the potential of the individual particle is gone.
            #2: upwind derivative. check velocity. use cell that you're moving towards...?
        
            #METHOD 1: np.gradient
            dx, dy, dz = np.gradient(potential)
            try:
                self.acc[0, ii] = -1*dx[x_idx]/self.m #divide by mass, which is just 1 by default
                self.acc[1, ii] = -1*dy[x_idx]/self.m
                self.acc[2, ii] = -1*dz[x_idx]/self.m
            except Exception as e:
                print(str(e))
                print('bruh it broke wait why??')
                print('x index is ', x_idx)
                print('num is', num)
                print('ii is', ii)
                print('x half is ', self.x_half)
                print('step number:  ', self.steps_taken)

                assert 1==0
            
            if DEBUG:
                mpl.figure()
                mpl.imshow(dx[0, :, :], cmap='hot')
                mpl.title('full gradient, x=0')
                mpl.colorbar()
                mpl.show()
        
            #METHOD 2: upwind derivative
            #check velocity to see which cell to differentiate.
            #will need guard cells for non-periodic.
            #will need to wrap around for periodic.
#             if self.v[0, ii] > 0:
#                 dx =  self.v[0, ii] - self.v[0, ii+1]
                
            
        #FOR EACH PARTICLE
            #need to get the single particles effect.
            #really just shifted version of green's potential
            #shifted so that the center is AT the particle.
            #then subtracted off.
            #probably use some mods or soemthing? could multiply in frequency space.
                #but that would be less efficient. just shift with matrix magic or sth.
            #will be bigger/smaller for periodic/nonperiodic so watch out for that.   
        return 0
    
    #update the grid so we know how many particles per grid cell
    def update_grid(self):
        #sum ceil of positions as x/y grids.
        #toggle between period and nonperiodic boundary conditions
        #test without changing velocity and position
        self.grid[:, :, :] = 0 #CLEAR GRID
        
        #determine which particles are still inside grid
        #CUTOFF VERSION
        idx = (self.x>0) & (self.x<self.ngrid)
        idx_inside = idx[0, :] & idx[1, :] & idx[2, :]
        x_inside = self.x[:, idx_inside]
        
        #turn the positions into grid indices
        x_grid_pos = tuple(map(tuple, x_inside.astype(int))) #MADE INTO INDICES
        np.add.at(self.grid, x_grid_pos, 1) #UPDATE GRID
        return 0

    def update_grid_half(self):
        #COPIED CODE FROM UPDATE_GRID but using self.x_half now.
        #yeah... I realize I could just PASS in the data... or use toggle...
        self.grid[:, :, :] = 0 #CLEAR GRID
        
        #determine which particles are still inside grid
        #CUTOFF VERSION
        idx = (self.x_half>0) & (self.x_half<self.ngrid)
        idx_inside = idx[0, :] & idx[1, :] & idx[2, :]
        x_inside = self.x_half[:, idx_inside]
        
        #turn the positions into grid indices
        x_grid_pos = tuple(map(tuple, x_inside.astype(int))) #MADE INTO INDICES
        np.add.at(self.grid, x_grid_pos, 1) #UPDATE GRID
        return 0
    
    
    def plot_density_heatmap(self):
        mpl.imshow(self.grid[0, :, :], cmap='hot')
        mpl.title('grid, x=0')
        mpl.show()
        
        mpl.imshow(self.grid[:, 0, :], cmap='hot')
        mpl.title('grid, y=0')
        mpl.show()
        
        mpl.imshow(self.grid[:, :, 0], cmap='hot')
        mpl.title('grid, z=0')
        mpl.show()
        
        return 0
        
    def take_step(self, DEBUG=False):
        #FOR A SINGLE PARTICLE
        #take the GRADIENT of the potential at its position
        #gives you the FORCE.
        #get the acceleration.
        #update the VELOCITY.
        #update the POSITION.
        #leapfrog somehow.
        
        #LEAP FROGGOOO
        #OK SO:
        #take x halfway into interval with currently estimated velocity
        #use THAT value of x to calculate the forces (so THAT'S when you'd call get potential I think. but would have to do for ALL of the particles
        #use THAT calculated force to get the velocity in the middle of the interval
        #use THAT velocity to update the x for real with a full timestep
        #then use the same force to update the velocity for real with a full timestep.
        
        #QUESTION: put ALL particles halfway out of position before calculating the potential? or just the one particle. probably all of the particles
        
#         self.x += self.dt*self.v
#         self.steps_taken += 1
        
        
        #TO DO:
        #take X halfway into interval with currently estimated velocity (HALF timestep)
        # use THAT value of the x to calculated the...
            #potential
            #gradient of potential: force
            #new velocity in the MIDDLE of the interval (with HALF timestep)
        #update X: use that velocity in the middle of the interval, full timestep
        #update V: use FULL timestep now
        
        #get halfway done x values. update grid and potential based on them.
        self.x_half = self.x + 1/2*self.dt*self.v
        self.update_grid_half() #updates grid to values of half x
        self.get_potential(DEBUG=DEBUG)
        
        #calculate the acceleration based on halfway x positions and potential
        self.calculate_acceleration(DEBUG=DEBUG)
        
        #calculate new halfway velocity in the middle of the interval
        self.v_half = self.v + 1/2*self.dt*self.acc
        
        #use that to fully update update x and v.
        self.x = self.x + self.dt*self.v_half
        self.v = self.v + self.dt*self.acc
        
        #update the grid with the new legit x positions
        self.update_grid()
        
        self.steps_taken +=1
        return 0
    
    def take_n_steps(self, n, plots=False, plot_every=10, DEBUG=False):
        for ii in range(0, n):
            self.take_step(DEBUG=DEBUG)
            if DEBUG:
                print('steps done: ', self.steps_taken)
            
            #plot positions every.... 10 steps
            if plots and (ii%plot_every == 0):
                self.plot_positions()
        return self.steps_taken
        
    def plot_positions(self):
        fig=mpl.figure(figsize=(10,10))#Create 3D axes
        try: 
            ax=fig.add_subplot(111,projection="3d")
        except : ax=Axes3D(fig) 
        ax.scatter(self.x[0], self.x[1], self.x[2],color="royalblue",marker="*", s=0.9)#,s=.02)
#         ax.scatter(self.x[0], self.x[1], self.x[2],color="royalblue",marker=".", s=0.9)#,s=.02)
        ax.set_xlabel("x-coordinate",fontsize=14)
        ax.set_ylabel("y-coordinate",fontsize=14)
        ax.set_zlabel("z-coordinate",fontsize=14)
        ax.set_title("Particle Positions\n",fontsize=20)
        # ax.legend(loc="upper left",fontsize=14)
        # ax.xaxis.set_ticklabels([])
        # ax.yaxis.set_ticklabels([])
        # ax.zaxis.set_ticklabels([])
        
        #put invisible points at the corners to hold the view steady
        ax.scatter(0, 0, 0, alpha=0)
        ax.scatter(0, self.ngrid, 0, alpha=0)
        ax.scatter(0, 0, self.ngrid, alpha=0)
        ax.scatter(0, self.ngrid, self.ngrid, alpha=0)
        ax.scatter(self.ngrid, 0, 0, alpha=0)
        ax.scatter(self.ngrid, self.ngrid, 0, alpha=0)
        ax.scatter(self.ngrid, 0, self.ngrid, alpha=0)
        ax.scatter(self.ngrid, self.ngrid, self.ngrid, alpha=0)

        mpl.savefig(self.outdir+'nbodystep'+str(self.steps_taken)+'.png', dpi=1200)
    

    def run_nbody(self, iters=10):
        #start the first plot
        return 0
        