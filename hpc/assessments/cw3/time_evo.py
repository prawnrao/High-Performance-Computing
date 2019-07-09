def time_evo(u):
    """ Function that takes in u, and returns u for the following timestep
        
    """
    
    kernel ="""
            __kernel void evolve(__global double *pos_eq_sol,
                                 __global double *curr_u,
                                 __global double *next_u)
            {
                int N = get_global_size(0);
                double dt = 1.0/(2*N*N);
                
                int x_id = get_global_id(0);
                int y_id = get_global_id(1);
                
                next_u[x_id * N + y_id] = pos_eq_sol[x_id * N + y_id] * dt +
                                          curr_u[x_id * N + y_id];
            
            }
            """
    
    # Builds the program from the kernel
    prg = cl.Program(ctx, kernel)
    prg.build()
    
    # Names the function from the program
    evolve_kernel = prg.evolve
    mf = cl.mem_flags
    
    u = np.ascontiguousarray(u)
    
    # Solves the differential of u, using the poisson equation
    pos_eq_sol = solve_poisson_equation(u).reshape(N,N)
    
    # Defines the buffers used by OpenCL
    pos_eq_buffer = cl.Buffer(ctx,mf.COPY_HOST_PTR | mf.READ_ONLY,
                              hostbuf = pos_eq_sol)
    
    curr_u_buffer = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY,
                              hostbuf = u)
    
    next_u_buffer = cl.Buffer(ctx, mf.ALLOC_HOST_PTR, size= (N*N*8))
    
    # Calls the funtion, using the queue and the buffers
    evolve_kernel(queue, (N, N), (1, 1), pos_eq_buffer, curr_u_buffer,
                  next_u_buffer)
    # Assigns the value for the next u from the buffer to the variable
    next_u, _ = cl.enqueue_map_buffer(queue, next_u_buffer, cl.map_flags.READ,
                                      0, (N*N,), np.double)
    
    queue.finish()
    
    return next_u
