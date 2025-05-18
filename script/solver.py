import os
import time
import numpy as np
import scipy 
import sympy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
import importlib

def main_solver(Result_dir, E, nu, num_cores, print_func):

    print_func("Solver running...")

    fn_par = importlib.import_module('script.Functions')

    E = float(E)  # Modulo di Young
    nu = float(nu)  # Coefficiente di Poisson
    num_cores = int(num_cores)  # Numero di processori da utilizzare

    # IMPORT MESH

    # Ottieni il percorso della cartella in cui si trova lo script
    input_dir = os.path.join(Result_dir, 'Results', 'Input')
    output_dir = os.path.join(Result_dir, 'Results', 'Output')

    # Costruisci il percorso relativo al file
    mesh_T_path = os.path.join(input_dir, 'mesh_T.txt')
    mesh_P_path = os.path.join(input_dir, 'mesh_P.txt')

    try:
        # Specifica il delimitatore come una virgola
        T_matrix = np.loadtxt(mesh_T_path, delimiter=',').T
        P_matrix = np.loadtxt(mesh_P_path, delimiter=',').T
    except Exception as e:
        print_func(f"Error while loading the files: {e}")
        return

    elapsed_time = np.zeros(6)  # Inizializza un array per il tempo di esecuzione
    # Inizializza il timer
    start_time = time.time()

    n_elems = np.size(T_matrix, 1)
    n_nodes = np.size(P_matrix, 1)


    # Assembly Matrice di Rigidezza K 
    try:
        # Passa il numero di processori alla funzione
        K = fn_par.K_global_assembly(P_matrix, T_matrix, E, nu, num_procs=num_cores)
    except Exception as e:
        print_func(f"Error during the assembly of the stiffness matrix: {e}")
        return
    
     # COMPUTATION TIME
    end_time = time.time()
    elapsed_time[0] = end_time - start_time
    print_func("Time taken for the assembly of the stiffness matrix: ", elapsed_time[0])


    start_time = time.time()

    # Imposizione Neumann BC
    f = fn_par.f_assembly(P_matrix, T_matrix,  input_dir, num_procs=num_cores)

     # COMPUTATION TIME
    end_time = time.time()
    elapsed_time[1] = end_time - start_time
    print_func("Time taken for the imposition of Neumann BC: ", elapsed_time[1])

    '''Potrei provare a fare in modo che venga calcolata in F una matrice di 
       connettività nodo-->matrice in modo da poter direttamente sommare i contributi
       delle forzanti unicamente sui nodi interessati. In questo modo eviterei di dover 
       controllare nel ciclo for tutti gli elementi. In questo caso probabilmente non sarebbe
       rendere il processo parallelo o quanto meno modifcarlo in modo che lo faccia in paralello
       per ogni nodo forzato e non per tutti gli elementi.'''

    start_time = time.time()

    # Applicazione Dirichlet BC
    K, f, nodi_vincolati = fn_par.vincolo_x(K, f, P_matrix, input_dir)

     # COMPUTATION TIME
    end_time = time.time()
    elapsed_time[2] = end_time - start_time
    print_func("Time taken for the imposition of Dirichlet BC: ", elapsed_time[2])

    start_time = time.time()

    # Calcolo spostamenti punti non vincolati
    toll = 1e-6
    u_new = fn_par.lin_syst_solver_par(K,f)
    print_func('Linear system resolution completed')

    end_time = time.time()
    elapsed_time[3] = end_time - start_time
    print_func("Time taken for the resolution of the linear system: ", elapsed_time[3])
    
    start_time = time.time()

    # Ricompongo il vettore u aggiungendo gli spostamenti(nulli) dei punti vincolati
    u = fn_par.u_composition(u_new, nodi_vincolati, n_nodes)

    end_time = time.time()
    elapsed_time[4] = end_time - start_time
    print_func("Time taken for the recomposition of the u vector: ", elapsed_time[4])

    start_time = time.time()

    # Calcolo sforzo elementi
    sigma, stress_vm, stress_vm_nodi = fn_par.sigma_assembly(P_matrix, T_matrix, u, E, nu, num_procs=num_cores)

    end_time = time.time()
    elapsed_time[5] = end_time - start_time
    print_func("Time taken for stress calculation: ", elapsed_time[5])

    # Ricalcolo la posizione dei nodi una volta calcolata la deformazione
    P_new = np.zeros((3, n_nodes))
    P_new[0, :] = P_matrix[0, :] + u[0::3]
    P_new[1, :] = P_matrix[1, :] + u[1::3]
    P_new[2, :] = P_matrix[2, :] + u[2::3]

    # COMPUTATION TIME
    print_func("Total time: ", sum(elapsed_time))

    # Costruisci il percorso completo per i file di output
    T_path = os.path.join(output_dir, 'mesh_T.txt')
    P_new_path = os.path.join(output_dir, 'P_new.txt')
    stress_vm_path = os.path.join(output_dir, 'stress_vm.txt')
    stress_vm_nodi_path = os.path.join(output_dir, 'stress_vm_nodi.txt')
    u_nodes_path = os.path.join(output_dir, 'u_nodes.txt')

    # Salva i file di output nella cartella Output
    np.savetxt(T_path, T_matrix, delimiter=',')
    np.savetxt(P_new_path, P_new, delimiter=',')
    np.savetxt(stress_vm_path, stress_vm, delimiter =',')
    np.savetxt(stress_vm_nodi_path, stress_vm_nodi, delimiter =',')
    np.savetxt(u_nodes_path, u, delimiter =',')

    print_func(f"Files saved in the folder: {output_dir}")

