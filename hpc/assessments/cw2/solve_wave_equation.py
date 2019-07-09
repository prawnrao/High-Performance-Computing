import numpy as np
import pyopencl as cl
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)


def solve_wave_equation(N, time_step, c=1, T=1):
    """OpenCL Implementation

    Solves a 1d wave equation.
    This function solves the 1d wave equation
    with zero boundary conditions. The x-domain is the
    interval [0, 1] and the final time is T.
    """
    # Defines the xt matrix and the x array
    xt_grid = np.zeros((N, time_step + 1), dtype='float64')
    xx = np.linspace(0, 1, N)

    # Calculates all the x values at t = 0 for the xt matrix
    xt_grid[:, 0] = np.exp(-5 * (xx - 0.5) ** 2)

    # Calculates the finite differences of x and t
    dt = T / (1.0 * time_step)
    dx = 1. / (N - 1)

    # Calculates the Courant number
    C = c * dt / dx

    kernel = """
    __kernel void solve_wave_equation(const int time_step,
                                      const int N,
                                      const double C,
                                      const int t_id,
                                      __global double *xt_grid,
                                      __global double *result_buffer)
    {
        int x_id = get_global_id(0);
        
        // Defines the index to the left of the current index
        int il = x_id - 1;
        if (x_id == 0) {
            il = 1;
        }
        
        // Defines the index to the right of the current index
        int ir = x_id + 1;
        if (x_id == (N - 1)) {
            ir = N - 2;
        }
        
        if (t_id == 0) {
            result_buffer[x_id] = xt_grid[x_id * (time_step)] -
                                  0.5 * C* C *(xt_grid[ir * (time_step)] -
                                  2 * xt_grid[x_id * (time_step)] +
                                  xt_grid[il * (time_step) ]);
        } else {
            result_buffer[x_id] = (- xt_grid[(time_step) * x_id + t_id - 1] +
                                  2 * xt_grid[(time_step) * x_id + t_id] +
                                  (C * C * (xt_grid[(time_step) * ir + t_id] -
                                  2 * xt_grid[(time_step) * x_id + t_id] +
                                  xt_grid[(time_step) * il + t_id])));
        }
    }

    """

    prg = cl.Program(ctx, kernel)
    prg.build()
    wave_equation_kernel = prg.solve_wave_equation
    mf = cl.mem_flags

    # Loops over all the time steps and solves the wave equation
    for t_id in range(0, time_step):
        # Assigns the buffer to the grid
        xt_grid_buffer = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY,
                                   hostbuf=xt_grid)

        # Assigns the buffer for the result
        result_buffer = cl.Buffer(ctx, mf.ALLOC_HOST_PTR,
                                  size=(N) * 8)

        # Calls the function defined in the kernel
        wave_equation_kernel(queue, (N, ), (1, ), np.int32(time_step + 1),
                             np.int32(N), np.float64(C), np.int32(t_id),
                             xt_grid_buffer, result_buffer)

        # Places the result into the matrix
        xt_grid[:, t_id + 1] = cl.enqueue_map_buffer(queue, result_buffer,
                                                     cl.map_flags.READ, 0,
                                                     (N, ), np.double)[0]
    queue.finish()

    return xt_grid


from time import time

T = 2
st = time()
res = solve_wave_equation(200, 500, c=1, T=T)
et = time()
print("Elapsed time (s): {0}".format(et - st))

plt.imshow(res, extent=[0, T, 0, 1])
plt.colorbar()
plt.show()
