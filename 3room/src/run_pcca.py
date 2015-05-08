import pickle
from pcca import PCCA

f = open('tmatrixperfect.dat','r')
a = pickle.Unpickler(f)
tm = a.load()

pccaobj = PCCA()
chi_mat = pccaobj.pcca(tm)

outfile = open('chi_mat.dat','w')
pickle.dump(chi_mat,outfile)