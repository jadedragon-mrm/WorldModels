import cupy as cp

x = cp.arange(6, dtype='f').reshape(2, 3)
y = cp.arange(3, dtype='f')

print(x)

kernel = cp.ElementwiseKernel(
    'float32 x, float32 y', 'float32 z',
    '''if (x - 2 > y) {
       z = x * y;
     } else {
       z = x + y;
     }''', 'my_kernel')

print(kernel(x, y))
