# coding=utf-8


import soundfile as sf
import sys 
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../build/'))

from python_pesq import pesq

ref, fs = sf.read(sys.argv[1])
deg, fs = sf.read(sys.argv[2])

print("{:.3f}".format(pesq(ref, deg, fs)))

