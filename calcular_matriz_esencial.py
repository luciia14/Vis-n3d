import numpy as np
def calcular_matriz_esencial(F, K):
   
    #Cálculamos la matriz esencial a traves de la ecuación E=K' F K 
    E = np.dot(np.dot(K.T, F), K)
    
    print("Matriz Esencial E:")
    print(E)
    
    return E

