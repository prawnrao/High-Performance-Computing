__kernel void solve_poisson_equation(__global double *u,
                                     __global double *sigma,
                                     __global double *solution,
                                     __global double *next_u)
{
    int N = get_global_size(0);
    double h = (double) 1.0/(N+1);
    double dt = (double) 1.0/(h*h);
    
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    // 5 Point Stencil for u and sigma
    
    double u_up; double u_down; double u_right; double u_left; double u_centre;
    double sigma_up; double sigma_down; double sigma_right; double sigma_left;
    double sigma_centre;
    
    if (i == 0) {
        u_up = 0;
        sigma_up = sigma[(i + 1) * N + j];
    } else {
        u_up = u[(i - 1) * N + j];
        sigma_up = sigma[(i - 1) * N + j];
    }
    
    if(j == 0) {
        u_left = 0;
        sigma_left = sigma[i * N + j + 1];
    }else {
        u_left = u[i * N + j - 1];
        sigma_left = sigma[i * N + j - 1];
    }
    
    if(j == (N - 1)) {
        u_right = 0;
        sigma_right = sigma[i * N + j - 1];
    }else {
        u_right = u[i * N + j + 1];
        sigma_right = sigma[i * N + j + 1];
    }
    
    if(i == (N - 1)){
        u_down = 0;
        sigma_down = sigma[(i - 1) * N + j];
    }else {
        u_down = u[(i + 1) * N + j];
        sigma_down = sigma[(i + 1) * N + j];
    }
    
    u_centre = u[i * N + j];
    sigma_centre = sigma[i * N + j];
    
    solution[i * N + j] = ((0.5 * (sigma_down + sigma_centre) * (u_down - u_centre)/h -
                            0.5 * (sigma_up + sigma_centre) * (u_centre - u_up)/h)/h +
                           (0.5 * (sigma_right + sigma_centre) * (u_right - u_centre)/h -
                            0.5 * (sigma_left + sigma_centre) * (u_centre - u_left)/h)/h);
    
    next_u[i * N + j] = pos_eq_sol[i * N + j] * dt + curr_u[i * N + j];
}
