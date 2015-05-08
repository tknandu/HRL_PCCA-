import pickle
import numpy as np

f = open('phi_mat.dat','r')
unpickler = pickle.Unpickler(f)
phi_mat = unpickler.load()

f = open('u_mat.dat','r')
unpickler = pickle.Unpickler(f)
u_mat = unpickler.load()

f = open('peeyush_u_mat.dat','r')
unpickler = pickle.Unpickler(f)
pu_mat = unpickler.load()

print phi_mat
print phi_mat[np.nonzero(phi_mat)]
