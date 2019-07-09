def evaluate_field_opencl(evaluation_points, source_positions,
                          strength_vector):
    """OpenCL implementation of the evaluate_field function
    """
    
    kernel = """
    __kernel void evaluate_potential(const int N,
                                    __global double *evaluationPoints,
                                    __global double *sourcePoints,
                                    __global double *strengthVector,
                                    __global double *field)
    {
        //Identifies the thread
        int i = get_global_id(0);
        
        //Resets potential matrix element
        double potential = 0.0f;
        
        //Evaluates the x and y values required to calculate the norm
        for(int j = 0; j < N; j++)
        {
            double currX = evaluationPoints[2 * i] - sourcePoints[2 * j];
            double currY = evaluationPoints[2 * i + 1] - sourcePoints[2 * j + 1];
            
            //Evaluates the norm
            double norm = sqrt(pow(currX,2) + pow(currY,2));
            
            //Evaluates the potential
            potential -= strengthVector[j] * log(norm);
        }
        
        //Fills the field matrix with potentials
        field[i] = potential;
    }
    """
    evaluation_points = np.ascontiguousarray(evaluation_points)
    n_eval_points = np.int32(evaluation_points.shape[0])
    n_source_points = np.int32(source_positions.shape[0])
    
    mf = cl.mem_flags
    
    eval_buffer = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY,
                            hostbuf=evaluation_points)
    source_buffer = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY,
                              hostbuf=source_positions)
    strength_buffer = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY,
                                hostbuf=strength_vector)
    result_buffer = cl.Buffer(ctx, mf.ALLOC_HOST_PTR,
                              size=evaluation_points.shape[0] * 8)
                            
    prg = cl.Program(ctx, kernel)
    prg.build()
    potential_kernel = prg.evaluate_potential
    
    potential_kernel(queue, (n_eval_points, ), (1, ), n_source_points,
                     eval_buffer, source_buffer, strength_buffer,
                     result_buffer)
    
    field, _ = cl.enqueue_map_buffer(queue, result_buffer,
                                     cl.map_flags.READ, 0,
                                     (n_eval_points,), np.double)
    queue.finish()
    return field
