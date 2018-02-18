import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# a "board" is a 2d array with zeros on the boundary


def zero_boundary(Z):
    "Sets the boundary elements of the 2d array to zero."
    assert(Z.ndim == 2)
    Z[:,0] = 0
    Z[:,-1] = 0
    Z[0,:] = 0
    Z[-1,:] = 0
    return Z


def sum_neighbors(Z):
    """For each non-boundary element of Z, compute the
    sum of the neighboring elements. """
    N = np.zeros(Z.shape)
    N[1:-1,1:-1] = (Z[:-2, :-2] + Z[1:-1, :-2] + Z[2:,:-2]
                  + Z[:-2,1:-1]                + Z[2:,1:-1]
                  + Z[:-2,2:] +   Z[1:-1, 2:]  + Z[2:,2:])
    return N


def step_once(Z):
    "Update state Z according to the rules of Life."
    N = sum_neighbors(Z)
    underpopulated = (N < 2)
    overpopulated = (N > 3)
    born = (Z == 0) & (N == 3)
    Z[underpopulated | overpopulated] = 0
    Z[born] = 1
    return Z


def plot(Z):
    "Plot the state Z of the game."
    fig = plt.figure(figsize=(Z.shape[0]/dpi, Z.shape[1]/dpi))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False,
                      xticks=[], yticks=[])
    ax.imshow(Z, interpolation='nearest', vmin=0, vmax=1, cmap='Greens')
    fig.show()

    
def animate(Z, iters, interval=200, figsize=(5,5)):
    "Animate the development, from initial state Z."
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False,
                      xticks=[], yticks=[])
    im = ax.imshow(Z, interpolation='nearest', vmin=0, vmax=1, cmap='Greens')

    def update(frame):
        step_once(Z)
        im.set_data(Z)
        return im,

    anim = FuncAnimation(fig, update, frames=iters, interval=interval, blit=True)
    fig.show()
    # we need to keep a reference to the animation so that it
    # does not get GC'ed... this is a known matplotlib issue
    return anim


def glider_demo():
    "A demo with a single glider."
    glider = np.array([[0,0,0,0,0,0],
                       [0,0,1,0,0,0],
                       [0,0,0,1,0,0],
                       [0,1,1,1,0,0],
                       [0,0,0,0,0,0],
                       [0,0,0,0,0,0]])

    Z = np.zeros((100,100))
    Z[:6,:6] = glider
    return animate(Z,100)

def random_demo():
    "A demo with a random initial state."
    Z = np.random.randint(2,size=(100,100))
    zero_boundary(Z)
    return animate(Z,1000,interval=20)

def blinker_demo():
    "A demo with a blinker."
    Z = np.array([[0,0,0,0,0],
                  [0,0,0,0,0],
                  [0,1,1,1,0],
                  [0,0,0,0,0],
                  [0,0,0,0,0]])
    return animate(Z,20)

def random_interior_demo():
    "Random initial configuration, inside a larger domain."
    Z = np.zeros((200,200))
    Z[50:150,50:150] = np.random.randint(2,size=(100,100))
    return animate(Z,1000,interval=20)

def weird_shape_demo():
    "A little shape that grows into a big one."
    Z = np.zeros((100,100))
    Z[48:51,48:51] = np.array([
        [0,1,0],
        [1,1,1],
        [0,0,1]])
    return animate(Z,1000,interval=20)
