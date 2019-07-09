import numpy as np
import pyopencl as cl
from scipy.sparse.linalg import LinearOperator, gmres, bicgstab
import matplotlib.pyplot as plt
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

def make_random_sigma(N, mean=2, sd=0.5, seed=0):
    S = np.random.RandomState(seed).normal(mean, sd, (N, N))
    sigma = np.exp(np.negative(S))
    return sigma

def make_sigma(N):
    points = np.linspace(0,1,N)
    
    kernel ="""
    __kernel void make_sigma(__global double *points,
                             __global double *sigma)
    {
        int N = get_global_size(0);
        
        int i = get_global_id(0);
        int j = get_global_id(1);

        sigma[i * N + j] = 1 + points[i]*points[i] + points[j]*points[j];
    }
    """
    
    prg = cl.Program(ctx, kernel)
    prg.build()
    make_sigma_kernel = prg.make_sigma
    mf = cl.mem_flags
    
    points_buffer = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=points)
    sigma_buffer = cl.Buffer(ctx, mf.ALLOC_HOST_PTR, size = N * N * 8)
    
    make_sigma_kernel(queue, (N, N), (1, 1), points_buffer, sigma_buffer)
    
    sigma, _ = cl.enqueue_map_buffer(queue, sigma_buffer, cl.map_flags.READ, 0, (N*N,), np.double)
    
    queue.finish()
    
    return sigma

def poisson_equation(u):
    """Matrix free opencl implementation to solve poisson equation.
    """
    
    kernel = """
        __kernel void solve_poisson_equation(__global double *u,
                                             __global double *sigma,
                                             __global double *result)
        {
            int N = get_global_size(0);
            double h = (double) 1.0/(N+1);
            
            int i = get_global_id(0);
            int j = get_global_id(1);
            
            // 5 Point Stencil for u and sigma
            
            double u_up; double u_down; double u_right; double u_left; double u_centre;
            double sigma_up; double sigma_down; double sigma_right; double sigma_left; double sigma_centre;
            
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
            
            result[i * N + j] = -1.0 * ((0.5 * (sigma_down + sigma_centre) * (u_down - u_centre)/h -
                                0.5 * (sigma_up + sigma_centre) * (u_centre - u_up)/h)/h +
                                (0.5 * (sigma_right + sigma_centre) * (u_right - u_centre)/h -
                                0.5 * (sigma_left + sigma_centre) * (u_centre - u_left)/h)/h);
            
        }

        """

    prg = cl.Program(ctx, kernel)
    prg.build()
    poisson_equation_kernel = prg.solve_poisson_equation
    mf = cl.mem_flags
    
    sigma_buffer = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_ONLY, hostbuf=sigma)
    u_buffer = cl.Buffer(ctx, mf.COPY_HOST_PTR | mf.READ_WRITE, hostbuf=u)
    result_buffer = cl.Buffer(ctx, mf.ALLOC_HOST_PTR, size = N * N * 8)

    poisson_equation_kernel(queue, (N, N), (1, 1), u_buffer, sigma_buffer, result_buffer)
    
    result, _ = cl.enqueue_map_buffer(queue, result_buffer, cl.map_flags.READ, 0, (N*N,), np.double)

    queue.finish()

    return result


N = 70
# sigma = make_sigma(N)
sigma = make_random_sigma(N)

A = LinearOperator((N*N,N*N),matvec=poisson_equation)
f = np.float64(np.ones(N*N))

residuals_gmres = []
def res_gmres(rk):
    res = np.linalg.norm(rk)
    residuals_gmres.append(res)

u_gmres, info_gmres = gmres(A, f, callback = res_gmres)
residuals_gmres = np.array(residuals_gmres)
u_gmres_mat = u_gmres.reshape(N,N)

u_gmres = np.zeros((N+2,N+2))
for i in range(1,N+1):
    for j in range(1,N+1):
        u_gmres[i][j] = u_gmres_mat[i-1][j-1]

plt.figure(dpi=200,figsize=(5,5))
plt.imshow(u_gmres,extent=[0,1,0,1],origin="lower")

residuals_bicgstab = []
def res_bicgstab(xk):
    res = np.linalg.norm(f - A*xk)
    residuals_bicgstab.append(res/np.linalg.norm(f))

u_bicgstab , info_bicgstab = bicgstab(A,f, callback = res_bicgstab)
residuals_bicgstab = np.array(residuals_bicgstab)
u_bicgstab_mat = u_bicgstab.reshape(N,N)

u_bicgstab = np.zeros((N+2,N+2))
for i in range(1,N+1):
    for j in range(1,N+1):
        u_bicgstab[i][j] = u_bicgstab_mat[i-1][j-1]

plt.figure(2,dpi=200,figsize=(5,5))
plt.imshow(u_bicgstab,extent=[0,1,0,1],origin="lower")

import matplotlib.pyplot as plt
%matplotlib inline

x_gmres = np.arange(1,len(residuals_gmres)+1)
x_bicgstab = np.arange(1,len(residuals_bicgstab)+1)

plt.figure(2,dpi=100,figsize=(8,5))
plt.semilogy(x_gmres, residuals_gmres, "b-", label="gmres")
plt.semilogy(x_bicgstab, residuals_bicgstab, "r-", label="bicgstab")
plt.ylabel("Norm of Residuals")
plt.xlabel("Number of Iterations")
plt.legend()
plt.grid(linestyle="--")

plt.show()

iterations_gmres = []
iterations_bicgstab = []
Npoints = []

for N in range(10,100,10):
    Npoints.append(N)
    residuals_gmres = []
    sigma = make_sigma(N)
    A = LinearOperator((N*N,N*N),matvec=poisson_equation,)
    f = np.float64(np.ones(N*N))
    u_gmres, info_gmres = gmres(A, f, callback = res_gmres)
    u_bicgstab , info_bicgstab = bicgstab(A,f, callback = res_bicgstab)
    iterations_gmres.append(len(residuals_gmres))
    iterations_bicgstab.append(len(residuals_bicgstab_bicgstab))

iterations_gmres = np.array(iterations_gmres)
iterations_bicgstab = np.array(iterations_bicgstab)
Npoints = np.array(Npoints)

plt.figure(dpi=200,figsize=(8,5))
plt.xlabel("Number of discretisation points")
plt.ylabel("Number of iterations for convergence")
plt.grid(linestyle="--")
plt.plot(Npoints,number_of_iterations,"x")
