
# An easy python's wrapper for pesq with pybind11

## How to use 
### 1 clone current project and cd it 
```bash
git clone https://github.com/huyanxin/python_pesq_with_pybind11/
cd python_pesq_with_pybind11
```

### 2. clone pybind11 or ln -s already exit dir

```bash 
git clone  https://github.com/pybind/pybind11.git
```
Or 

```bash 
ln -s your_exit_pybind11_dir .
```

### 3. Compile project 
```bash 
mkdir build && cd build
cmake ..
make 
```
### 4. use it 

```python

import soundfile as sf
import sys
sys.path.append(the pwd of the python_pesq.cpython-36m-x86_64-linux-gnu.so)
from python_pesq import pesq

ref, fs = sf.read('ref.wav')
deg, fs = sf.read('deg.wav')

print(pesq(ref,deg,fs))

```

### Thanks And Reference
[pybind11](https://github.com/pybind/pybind11.git)

[ITU P.862](https://www.itu.int/rec/T-REC-P.862-200102-I/en)

[vBaicai's python-pesq](https://github.com/vBaiCai/python-pesq)
