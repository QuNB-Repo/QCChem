#!/usr/bin/env python
# coding: utf-8

# In[5]:


from qiskit import Aer
from qiskit.opflow import X, Z, I
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SLSQP
from qiskit.circuit.library import TwoLocal
import numpy as np


arn6 = []
for a in np.arange(0,1.01,0.01, dtype=object):
    b = 1-a
    op = a*((X ^I ^ I ^ I ^ I ^ I)+(I ^X ^ I ^ I ^ I ^ I) +(I ^I ^ X ^ I ^ I ^ I)+ (I ^I ^ I ^ X^ I ^ I)+(I ^I ^ I ^ I^ X ^ I)+(I ^I ^ I ^ I^ I ^ X))+ b*((Z^Z^I^I^I^I)+(I^Z^Z^I^I^I) + (I^I^Z^Z^I^I)+(I^I^I^Z^Z^I)+(I^I^I^I^Z^Z)+(Z^I^I^I^I^Z))
    seed = 50
    algorithm_globals.random_seed = seed
    qi = QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)

    layer_1 = [(0, 5),(1,4),(2,3)]
    layer_2 = [(0, 3),(1,2),(4,5)]
 
    
    from qiskit.circuit.library import TwoLocal
    num_qubits = 6

    ansatz=TwoLocal(num_qubits, 'ry', 'cx',[layer_1,layer_2]
                      , reps=6,insert_barriers=True, parameter_prefix = 'theta')
    
    
    
    slsqp = SLSQP(maxiter=200)
    vqe = VQE(ansatz, optimizer=slsqp, quantum_instance=qi)
    result = vqe.compute_minimum_eigenvalue(operator=op)
    #print(result)
    optimizer_evals = result.optimizer_evals
    print("a:",a)
    print(">")
    #print("REsult:", result)
    print("")
    print("======================================***=============================================")
    arn6.append(result.eigenvalue.real)
print(arn6)



print("The circuit depth is:", ansatz.decompose().depth())
(ansatz.decompose().draw(output = 'mpl'))


# In[ ]:




