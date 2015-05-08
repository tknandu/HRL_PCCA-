import pickle

f = open('chi_mat.dat','r')
a = pickle.Unpickler(f)
chi_mat = a.load()

print chi_mat
print chi_mat.shape
