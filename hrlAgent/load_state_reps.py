import pickle
import numpy as np

def getBins(i_array,n_bins = 10):
    bins = [0.0]
    i_array.sort()

    i = 0
    while i < len(i_array):
        bins.append(i_array[i])
        i += len(i_array)/n_bins
    bins.append(1.0)
    return bins

f = open('state_reps.dat','r')
a = pickle.Unpickler(f)
state_reps = a.load()
new_state_reps = []
for state_rep in state_reps:
    new_state_reps.append([state_rep[0][0],state_rep[1][0],state_rep[2][0]])
new_state_reps = np.array(new_state_reps)

second_col = new_state_reps[:,1]
second_col_bins =getBins(second_col)
print second_col_bins

third_col = new_state_reps[:,2]
third_col_bins = getBins(third_col)
print third_col_bins


print len(second_col_bins)
print len(third_col_bins)
print (np.digitize([0.5],second_col_bins)[0],np.digitize([0.5],third_col_bins)[0])

