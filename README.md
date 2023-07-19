# HyRay
### `GPU ? CuPy : NumPy`
GPU arrays (via CuPy) when possible, otherwise CPU arrays (via NumPy)

The `hyray` library provides a single interface to use both NumPy and CuPy, depending on the availability of CuPy, CUDA, or sufficient memory. With `hyray` a single implementation can run with CuPy when CuPy is available (on a local machine with a GPU), and still work with NumPy when on GPU or insufficient memory is available (such as on continuous integration servers).

