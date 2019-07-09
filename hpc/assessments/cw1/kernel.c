"""
__kernel void evaluate_potential(const int N,
                                 __global double *evaluationPoints,
                                 __global double *sourcePoints,
                                 __global double *strengthVector,
                                 __global double *field)
{
    //Identifies the thread
    const int id = get_global_id(0);
    const int gid = TS*get_global_id(0) + id;
    
    //Local memory to fit a tile with dimensions TS
    __local double evalSub[TS][2];
    __local double sourceSub[TS][2];
    
    //Resets potential matrix element
    double potential = 0.0f;
    
    //Loop over all tiles
    const int nTs = N/TS;
    //Evaluates the x and y values required to calculate the norm
    for(int t = 0; t < nTs; t++)
    {
        const int xTid = TS*t;
        const int yTid = TS*t + 1;
        
        evalSub[id][0] = evaluationPoints[
        
        double currX = evaluationPoints[2 * id] - sourcePoints[2 * j];
        double currY = evaluationPoints[2 * id + 1] - sourcePoints[2 * j + 1];
        
        //Evaluates the norm
        double norm = sqrt(pow(currX,2) + pow(currY,2));
        
        //Evaluates the potential
        potential -= strengthVector[j] * log(norm);
    }
    
    //Fills the field matrix with potentials
    field[id] = potential;
}
"""
