---
title: Heterogeneous Parallel Computing
---
# FDTD
## calculate on a GPU
### code
|main.cu|kernel.u|kenel.cuh|
|-------|--------|---------|
|malloc memory、initialize variable、call the functions、free memory|calculate ez in one GPU|declare kernel function|

```cuda
main.cu
    main(): 
        define ntimestep = 200, row = 256 ,column =256
        caculate mem_size = sizeof(float) * row * coluumn 
        malloc memory for variable *h_hx1d, *h_hy1d, *h_ez1d
        use for loop to malloc dynamic memory for variable *h_hx2d, *h_hy2d, *h_ez2d
        use **cudaMalloc** to malloc memory for d_hx, d_hy, d_ez which size = mem_size
        use a for loop to initialize h_hx1d, h_hy1d, h_ez1d
        use cudaMemcpy to copy the host memory to device
        difine grid(16, 16, 1) --> gridDim.x = gridDim .y =16
        difine threads(16, 16, 1) --> blockDim.x = blockDim .y =16
        call the iteration function
        use cudaMemcpy to copy the device memory to host
        use a for loop to assign the data from the 1D array ez1d to the 2D array ez2d  
        free host memory and device memory  
    cpuiteration():
        Define source position: isource = 16*8+8, jsource = 16*8+8
        Dynamically allocate memory for hx, hy, ez
        Use nested loops to set boundary conditions
        Calculate electromagnetic fields according to FDTD equations
        Output results to the cpuez.dat file 
```
#### CUDA Kernel `iteration` Execution Process Analysis

##### Initialization and Parameter Setup

First, the kernel defines several physical constants and grid parameters:
- Physical constants: π, speed of light c, wavelength, etc.
- Spatial and temporal steps: dx and dt
- Angular frequency omega

##### Thread Index Calculation

Each thread calculates its position in the entire computational domain:
- Using blockIdx and threadIdx to determine the thread's position in the 2D grid
- Computing a unique global index `kn` for accessing array elements in global memory
- Defining the source position `source`

##### Computation Execution

The kernel performs two different calculations based on the `type` parameter:

###### When type=0 (Update Electric Field Ez):

1. **Boundary Condition Handling**:
   - If the thread is located at the boundary of the computational domain, Ez is set to 0
   
2. **Source Processing**:
   - If the current position is the source position, Ez is set to a sine wave: `sin(omega*n*dt)`
   
3. **FDTD Update**:
   - For other positions, Ez is updated according to the FDTD equation:
   ```
   Ez[kn] += 0.5*((Hy[kn]-Hy[kn-1])-(Hx[kn]-Hx[kn-Dim]))
   ```
   - This implements the update of the electric field Ez based on spatial derivatives of the magnetic fields Hx and Hy

###### When type=1 (Update Magnetic Fields Hx and Hy):

1. **Update Hx**:
   - If not at the upper boundary in the y direction, Hx is updated based on the spatial derivative of Ez:
   ```
   Hx[kn] -= 0.5*(Ez[kn+Dim]-Ez[kn])
   ```
   - Otherwise, Hx is set to 0 at the boundary

2. **Update Hy**:
   - If not at the right boundary in the x direction, Hy is updated based on the spatial derivative of Ez:
   ```
   Hy[kn] += 0.5*(Ez[kn+1]-Ez[kn])
   ```
   - Otherwise, Hy is set to 0 at the boundary
### questions
- 可不可以写成`float* h_hx1d =malloc(mem_size)`;没有`float*`会怎么样
```
    float* h_hx1d =(float*)malloc(mem_size);
```
在C语言中，`malloc`函数用于动态分配内存，并返回一个指向分配内存的指针。它的返回值是`void*`类型，这意味着它是一个通用指针，必须显式地转换为适当的指针类型才能正确使用。
- 为什么分配二维数组时，行可以像一维直接分配，列却需要使用循环
```cuda

    float ** h_hx2d=(float **)malloc(row*sizeof(float*) );
    float ** h_hy2d=(float **)malloc(row*sizeof(float*) );
    float ** h_ez2d=(float **)malloc(row*sizeof(float*) );
    for (i=0;i<row;i++){
        h_hx2d[i]=(float*)malloc(column*sizeof(float));
        h_hy2d[i]=(float*)malloc(column*sizeof(float));
        h_ez2d[i]=(float*)malloc(column*sizeof(float));
    }
```
在C语言中，二维数组的内存分配与一维数组有些不同，因为C语言本身不直接支持动态的二维数组。为了实现动态的二维数组，通常需要分两步进行内存分配：首先为行指针数组分配内存，然后为每一行分配内存。
- grid, theards参数定义
```cuda
dim3 grid(16,16,1);
dim3 threads(16,16,1);
```
在CUDA编程中，`dim3`类型用于定义网格（grid）和线程块（block）的维度。CUDA通过将计算任务分配到多个线程中并行执行来加速计算，而这些线程被组织成线程块和网格。
- 核函数调用
```cuda

    for (i=0;i<ntimestep;i++) {
        iteration <<< grid, threads >>>(d_hx,d_hy,d_ez,i,0);
        iteration <<< grid, threads >>>(d_hx,d_hy,d_ez,i,1);
        cudaThreadSynchronize();
    }
```
- 为什么要循环多次调用iteration
1. 时间步进：在电磁波传播模拟中，电场和磁场的状态需要在每个时间步上进行更新。每次迭代对应于模拟中的一个时间步。
2. 逐步更新场变量：电场（`g_ez`）和磁场（`g_hx`和`g_hy`）的值需要逐步更新，以模拟电磁波在空间中的传播。

- 这个函数中，kn是一个已知的定值吗？
`kn` 不是一个定值，而是一个动态计算的索引值，用于确定当前 CUDA 线程应该处理的数组元素位置。
```cuda
unsigned int kn = gridDim.x * blockDim.x * blockDim.y * blockIdx.y   
    + gridDim.x * blockDim.x * threadIdx.y
    + blockDim.x * blockIdx.x
    + threadIdx.x;
```
这个计算公式将二维网格中的线程位置映射到一维数组索引。

1. 每个线程执行时，都会基于自己的位置信息计算出唯一的 `kn` 值：
   - `blockIdx.x`, `blockIdx.y`: 当前线程块在网格中的坐标
   - `threadIdx.x`, `threadIdx.y`: 当前线程在块内的坐标
   - `gridDim.x`: 网格在 x 方向的块数量
   - `blockDim.x`, `blockDim.y`: 每个块在 x 和 y 方向的线程数量

2. 每个线程执行时会得到不同的 `kn` 值，对应于它负责处理的网格点

3. 这种计算方式实际上是将二维网格的坐标 (x, y) 映射到一维数组的索引，使得每个线程可以访问自己负责的数组元素

因此，`kn` 是一个变量，其值取决于执行该代码的特定线程在 CUDA 网格中的位置。每个线程都会计算出自己独特的 `kn` 值，用于访问全局内存中的相应数据元素。
- 比较两种索引方法
```cuda
    unsigned int Dim=gridDim.x*blockDim.x;
    unsigned int kn =gridDim.x * blockDim.x * blockDim.y * blockIdx.y   
        + gridDim.x*blockDim.x * threadIdx.y
        + blockDim.x * blockIdx.x
        + threadIdx.x;    
```
```cuda
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int index = row*width + col;
```
1. 表达方式不同
   - 第一种方法直接用一个公式计算
   - 第二种方法分步计算，先计算全局的行列索引，再转换为一维索引

2. width 参数的含义
   - 在第二种方法中，`width` 表示二维网格在 x 方向（列方向）的总宽度
   - 在第一种方法中，这个值隐含为 `gridDim.x * blockDim.x`

3. 假设条件
   - 第一种方法假设网格在 x 方向的总宽度就是 `gridDim.x * blockDim.x`
   - 第二种方法中的 `width` 可能是任意值（通常是数据的实际宽度）

4. 等价性证明
如果 `width = gridDim.x * blockDim.x`，那么两种方法是完全等价的：
- 时间步200，每个i都会执行65536个线程吗？
1. 循环执行 200 次（`i` 从 0 到 199）
2. 每次循环中：
   - 第一次核函数调用启动 65,536 个线程（更新电场）
   - 第二次核函数调用也启动 65,536 个线程（更新磁场）
   - `cudaThreadSynchronize()` 确保所有线程完成执行后再进入下一个时间步
- kernal边界条件
1. 电场为零的条件：
   - 在理想导体的表面，电场的切向分量必须为零。这是因为任何非零的切向电场分量都会在导体中感应出电流，直到电场消失。
   - 在代码中，通过将电场 `g_ez[kn]` 设置为零，确保了在仿真区域的边界上满足PEC条件。

2. 边界线程的处理：
   - 代码中条件 `((bx==0)&&(tx==0))||((bx==(gridDim.x-1))&&(tx==(blockDim.x-1)))||((by==0)&&(ty==0))||((by==(gridDim.y-1))&&(ty==(blockDim.y-1)))` 检查的是线程是否位于仿真区域的边界。
   - 如果线程位于边界位置，则将对应的电场值 `g_ez[kn]` 设置为零，模拟理想导体的反射特性。
- cpuiteration边界条件
1. 电场Ez的更新
- 循环范围是 `i=1` 到 `i=row-2`，`j=1` 到 `j=column-2`
- 这意味着边界上的电场值 `ez[0][j]`, `ez[row-1][j]`, `ez[i][0]`, `ez[i][column-1]` 不会被更新
- 这些边界值保持为初始值0，相当于实现了一个固定边界条件（Dirichlet边界条件）
2. 磁场Hx的更新
- 循环范围是 `i=0` 到 `i=row-1`，`j=0` 到 `j=column-2`
- 所有行的Hx都被更新，但最后一列的Hx不更新
- 这是因为Hx在x方向上与Ez交错排列，最后一列的Hx不需要计算
3. 磁场Hy的更新
- 循环范围是 `i=0` 到 `i=row-2`，`j=0` 到 `j=column-1`
- 所有列的Hy都被更新，但最后一行的Hy不更新
- 这是因为Hy在y方向上与Ez交错排列，最后一行的Hy不需要计算





