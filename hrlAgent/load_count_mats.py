import pickle
import numpy as np
import sys

n_eps = 2000 
f = open('phi_mat' + str(n_eps) + '.dat','r')
unpickler = pickle.Unpickler(f)
phi_mat = unpickler.load()

f = open('u_mat' + str(n_eps) + '.dat','r')
unpickler = pickle.Unpickler(f)
u_mat = unpickler.load()

f = open('peeyush_u_mat' + str(n_eps) + '.dat','r')
unpickler = pickle.Unpickler(f)
pu_mat = unpickler.load()


n_states = phi_mat.shape[0]
n_actions = phi_mat.shape[1]
#Using the matrices from page 39 of thesis
d_mat = np.zeros((n_states,n_states),dtype=float)
for state_1 in range(n_states):
    for state_2 in range(n_states):
        total = 0
        for action in range(n_actions):
            total += u_mat[state_1][action][state_2]
        d_mat[state_1][state_2] = float(total)

f = open('d_mat' + str(n_eps) + '.dat','w')
pickle.dump(d_mat,f)
p_mat = np.zeros((n_states,n_actions,n_states),dtype=float)
for state_1 in range(n_states):
    for action in range(n_actions):
        total = 0
        for state_2 in range(n_states):
            total += u_mat[state_1][action][state_2]
        if total == 0:
            p_mat[state_1][action][state_1] = 1 #If we have no transitions for (s_1,a), then set p(s_1,a,s_1) to 1 and let the others be 0
        else:
            for state_2 in range(n_states):
                p_mat[state_1][action][state_2] = u_mat[state_1][action][state_2]/float(total)

f = open('p_mat' + str(n_eps) + '.dat','w')
pickle.dump(p_mat,f)

t_mat = np.zeros((n_states,n_states),dtype=float)
for state_1 in range(n_states):
    total = 0
    for state_2 in range(n_states):
        total += d_mat[state_1][state_2]
    if total == 0:
        t_mat[state_1][state_1] = 1 #If we have no transitions out of s_1, then we set p(s_1,s_1) to 1 and let the others be 0
    else:
        for state_2 in range(n_states):
            t_mat[state_1][state_2] = d_mat[state_1][state_2]/float(total)

f = open('t_mat' + str(n_eps) + '.dat','w')
pickle.dump(t_mat,f)


#Now we dump the matrices computed using Peeyush's algo.
pd_mat = np.zeros((n_states,n_states),dtype=float)
for state_1 in range(n_states):
    for state_2 in range(n_states):
        total = 0
        for action in range(n_actions):
            total += pu_mat[state_1][action][state_2]
        pd_mat[state_1][state_2] = float(total)

f = open('peeyush_d_mat' + str(n_eps) + '.dat','w')
pickle.dump(pd_mat,f)
pp_mat = np.zeros((n_states,n_actions,n_states),dtype=float)
for state_1 in range(n_states):
    for action in range(n_actions):
        total = 0
        for state_2 in range(n_states):
            total += pu_mat[state_1][action][state_2]
        if total == 0:
            pp_mat[state_1][action][state_1] = 1 #If we have no transitions for (s_1,a), then set p(s_1,a,s_1) to 1 and let the others be 0
        else:
            for state_2 in range(n_states):
                pp_mat[state_1][action][state_2] = pu_mat[state_1][action][state_2]/float(total)

f = open('peeyush_p_mat' + str(n_eps) + '.dat','w')
pickle.dump(pp_mat,f)

pt_mat = np.zeros((n_states,n_states),dtype=float)
for state_1 in range(n_states):
    total = 0
    for state_2 in range(n_states):
        total += pd_mat[state_1][state_2]
    if total == 0:
        pt_mat[state_1][state_1] = 1 #If we have no transitions out of s_1, then we set p(s_1,s_1) to 1 and let the others be 0
    else:
        for state_2 in range(n_states):
            pt_mat[state_1][state_2] = pd_mat[state_1][state_2]/float(total)

f = open('peeyush_t_mat' + str(n_eps) + '.dat','w')
pickle.dump(pt_mat,f)

if len(sys.argv) > 1:
    if sys.argv[1].startswith("noreward"):
        d_mat = np.zeros((n_states,n_states),dtype=float)
        for state_1 in range(n_states):
            for state_2 in range(n_states):
                total = 0
                for action in range(n_actions):
                    total += phi_mat[state_1][action][state_2]
                d_mat[state_1][state_2] = float(total)

        f = open('noreward_d_mat' + str(n_eps) + '.dat','w')
        pickle.dump(d_mat,f)
        p_mat = np.zeros((n_states,n_actions,n_states),dtype=float)
        for state_1 in range(n_states):
            for action in range(n_actions):
                total = 0
                for state_2 in range(n_states):
                    total += phi_mat[state_1][action][state_2]
                if total == 0:
                    p_mat[state_1][action][state_1] = 1 #If we have no transitions for (s_1,a), then set p(s_1,a,s_1) to 1 and let the others be 0
                else:
                    for state_2 in range(n_states):
                        p_mat[state_1][action][state_2] = phi_mat[state_1][action][state_2]/float(total)

        f = open('noreward_p_mat' + str(n_eps) + '.dat','w')
        pickle.dump(p_mat,f)

        t_mat = np.zeros((n_states,n_states),dtype=float)
        for state_1 in range(n_states):
            total = 0
            for state_2 in range(n_states):
                total += d_mat[state_1][state_2]
            if total == 0:
                t_mat[state_1][state_1] = 1 #If we have no transitions out of s_1, then we set p(s_1,s_1) to 1 and let the others be 0
            else:
                for state_2 in range(n_states):
                    t_mat[state_1][state_2] = d_mat[state_1][state_2]/float(total)

        f = open('noreward_t_mat' + str(n_eps) + '.dat','w')
        pickle.dump(t_mat,f)
