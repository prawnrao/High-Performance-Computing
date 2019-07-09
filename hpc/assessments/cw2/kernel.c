__kernel void solve_wave_equation(const int time_step,
                                  const int N,
                                  const double C,
                                  const int t_id,
                                  __global double *xt_grid,
                                  __global double *result_buffer)
{
    int x_id = get_global_id(0);
    
    // Defines the index to the left of the current index
    int il = x_id + 1;
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
