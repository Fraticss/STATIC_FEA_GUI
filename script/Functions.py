import os
import numpy as np
from numpy.linalg import det
from scipy.sparse import lil_matrix

def D_assembly(E, nu):
    D = (E / ((1 + nu) * (1 - 2 * nu))) * \
        np.array([[1 - nu, nu, nu, 0, 0, 0],
                  [nu, 1 - nu, nu, 0, 0, 0],
                  [nu, nu, 1 - nu, 0, 0, 0],
                  [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
                  [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
                  [0, 0, 0, 0, 0, (1 - 2 * nu) / 2]])
    return D

def DN_assembly(coord_nodi):
    ''' 
    Nodi è una matrice 3x4 contenente le coordinate dei 4 nodi
    Ogni riga di nodi è del tipo [x_i, y_i, z_i] per il nodo i
    Output N è una cell array che contiene le funzioni di forma N1, N2, N3, N4 
    '''

    # Estrai le coordinate dei nodi
    x1 = coord_nodi[0, 0]; y1 = coord_nodi[1, 0]; z1 = coord_nodi[2, 0]
    x2 = coord_nodi[0, 1]; y2 = coord_nodi[1, 1]; z2 = coord_nodi[2, 1]
    x3 = coord_nodi[0, 2]; y3 = coord_nodi[1, 2]; z3 = coord_nodi[2, 2]
    x4 = coord_nodi[0, 3]; y4 = coord_nodi[1, 3]; z4 = coord_nodi[2, 3]

    # Costruisci la matrice di nodi (coefficienti per il sistema lineare)
    A = np.array([[1, x1, y1, z1],
                  [1, x2, y2, z2],
                  [1, x3, y3, z3],
                  [1, x4, y4, z4]])

    # Determinante della matrice A = Volume del tetraedro
    V = np.abs(det(A) / 6)
    if V == 0:
        print("Volume del tetraedro:", V)
        print(A)

    # Funzioni di forma (isoparametriche) N1, N2, N3, N4
    # Questi coefficienti derivano dalla risoluzione di un sistema lineare
    N1 = lambda x, y, z: det(np.array([
        [1, x, y, z],
        [1, x2, y2, z2],
        [1, x3, y3, z3],
        [1, x4, y4, z4]])) / (6 * V)
    
    N2 = lambda x, y, z: det(np.array([
        [1, x1, y1, z1],
        [1, x, y, z],
        [1, x3, y3, z3],
        [1, x4, y4, z4]])) / (6 * V)

    N3 = lambda x, y, z: det(np.array([
        [1, x1, y1, z1],
        [1, x2, y2, z2],
        [1, x, y, z],
        [1, x4, y4, z4]])) / (6 * V)

    N4 = lambda x, y, z: det(np.array([
        [1, x1, y1, z1],
        [1, x2, y2, z2],
        [1, x3, y3, z3],
        [1, x, y, z]])) / (6 * V)
    
    # Derivate parziali
    # Essendo le funzioni N lineari la dN1/dx = N1(x=1,y=0,z=0) ecc...
    zero1 = N1(0, 0, 0)
    zero2 = N2(0, 0, 0)
    zero3 = N3(0, 0, 0)
    zero4 = N4(0, 0, 0)
    Dp_N1 = [N1(1, 0, 0) - zero1, N1(0, 1, 0) - zero1, N1(0, 0, 1) - zero1]
    Dp_N2 = [N2(1, 0, 0) - zero2, N2(0, 1, 0) - zero2, N2(0, 0, 1) - zero2]
    Dp_N3 = [N3(1, 0, 0) - zero3, N3(0, 1, 0) - zero3, N3(0, 0, 1) - zero3]
    Dp_N4 = [N4(1, 0, 0) - zero4, N4(0, 1, 0) - zero4, N4(0, 0, 1) - zero4]
    Dp = np.array([Dp_N1, Dp_N2, Dp_N3, Dp_N4]).T

    return Dp

def B_assembly(Dp):
    B = np.zeros((6, 12))
    B[0, :] = [Dp[0][0], 0, 0, Dp[0][1], 0, 0, Dp[0][2], 0, 0, Dp[0][3], 0, 0]
    B[1, :] = [0, Dp[1][0], 0, 0, Dp[1][1], 0, 0, Dp[1][2], 0, 0, Dp[1][3], 0]
    B[2, :] = [0, 0, Dp[2][0], 0, 0, Dp[2][1], 0, 0, Dp[2][2], 0, 0, Dp[2][3]]
    B[3, :] = [Dp[1][0], Dp[0][0], 0, Dp[1][1], Dp[0][1], 0, Dp[1][2], Dp[0][2], 0, Dp[1][3], Dp[0][3], 0]
    B[4, :] = [0, Dp[2][0], Dp[1][0], 0, Dp[2][1], Dp[1][1], 0, Dp[2][2], Dp[1][2], 0, Dp[2][3], Dp[1][3]]
    B[5, :] = [Dp[2][0], 0, Dp[0][0], Dp[2][1], 0, Dp[0][1], Dp[2][2], 0, Dp[0][2], Dp[2][3], 0, Dp[0][3]]
    
    return B

def K_element_assembly(coord_nodi, E, nu):
    matrix = np.hstack((np.ones((4, 1)), coord_nodi.T))
    V = np.abs(1/6 * det(matrix))
    Dp = DN_assembly(coord_nodi)
    B = B_assembly(Dp)
    D = D_assembly(E, nu)
    K_local = (np.dot(B.T,np.dot(D,B))) * V
    return K_local


from multiprocessing import Pool

def assemble_partial(args):
    """Calcola una porzione di K_global per un sottoinsieme di elementi"""
    P, T, Elasticity, nu, elem_range = args
    n_nodi = P.shape[1]
    K_partial = lil_matrix((3 * n_nodi, 3 * n_nodi))

    for i in elem_range:
        nodi = T[:, i].astype(int)
        coord_nodi = P[:, nodi-1]
        K_local = K_element_assembly(coord_nodi, Elasticity, nu)

        index = np.array([3 * nodi[0] - 2, 3 * nodi[0] - 1, 3 * nodi[0],
                          3 * nodi[1] - 2, 3 * nodi[1] - 1, 3 * nodi[1],
                          3 * nodi[2] - 2, 3 * nodi[2] - 1, 3 * nodi[2],
                          3 * nodi[3] - 2, 3 * nodi[3] - 1, 3 * nodi[3]]) - 1

        ixgrid = np.ix_(index, index)
        K_partial[ixgrid] += K_local

    return K_partial

def K_global_assembly(P, T, Elasticity, nu, num_procs=4):
    """Parallelizza il calcolo di K_global usando multiprocessing"""
    n_elementi = T.shape[1]
    chunk_size = n_elementi // num_procs
    ranges = [range(i, min(i + chunk_size, n_elementi)) for i in range(0, n_elementi, chunk_size)]

    with Pool(processes=num_procs) as pool:
        K_partials = pool.map(assemble_partial, [(P, T, Elasticity, nu, r) for r in ranges])

    # Sommare tutte le matrici sparse parziali in una globale
    K_global = sum(K_partials, lil_matrix((3 * P.shape[1], 3 * P.shape[1])))
    
    return K_global.tocsr()


def AreaTriangolo(coord_nodi):
    # Estrae le coordinate dei nodi
    x1, y1, z1 = coord_nodi[:, 0]
    x2, y2, z2 = coord_nodi[:, 1]
    x3, y3, z3 = coord_nodi[:, 2]
    
    # Calcola l'area del triangolo usando la formula di Erone
    A = 0.5 * np.linalg.norm(np.cross([x2 - x1, y2 - y1, z2 - z1], [x3 - x1, y3 - y1, z3 - z1]))
    cg_x = (x1 + x2 + x3) / 3
    cg_y = (y1 + y2 + y3) / 3
    cg_z = (z1 + z2 + z3) / 3
    cg = np.array([cg_x, cg_y, cg_z])

    return A,cg

def f_assemble_partial(args):
    """Calcola una porzione di K_global per un sottoinsieme di elementi"""
    P, T, elem_range, input_dir = args
    n_nodi = P.shape[1]
    f_partial = np.zeros((3 * n_nodi,1))

    # Costruisci il percorso relativo al file
    Neumann_BC_path = os.path.join(input_dir, 'Neumann_BC.txt')
    
    Neumann_BC = np.loadtxt(Neumann_BC_path, delimiter=',')
    nodi_forzati = Neumann_BC[:,0].astype(int)
    cond_BC = np.array([Neumann_BC[:,1],Neumann_BC[:,2],Neumann_BC[:,3]]).T
    cond_BC = cond_BC.astype(int)
    stress_BC = np.array([Neumann_BC[:,4],Neumann_BC[:,5],Neumann_BC[:,6]]).T

    n_nodi = P.shape[1]
    n_elementi = T.shape[1]

    for i in elem_range:
        
        counter = 0
        nodi_faccia = np.array([])
        index = np.array([])
        
        for j in range(4):
            nodo = int(T[j,i])
            for k in range(np.size(nodi_forzati)):
                if nodo == nodi_forzati[k]:
                    counter += 1
                    nodi_faccia = np.append(nodi_faccia,nodo)
                    index = np.append(index,k)
        
        nodes_index = index.astype(int)

        if counter == 3:
            
            coord_nodi = P[:, nodi_faccia.astype(int)-1]
            
            Area, cg = AreaTriangolo(coord_nodi)

            for j in range(3):

                if cond_BC[nodes_index[0],j] == 1:
                
                    F_risultante = stress_BC[nodes_index,j] * Area
                    forza_nodo = (1 / 3) * F_risultante  # Forza sul nodo
                    #forze_nodi = forza_nodo * np.ones((3,1))  # Forze sui nodi
                    index = nodi_faccia * 3 - 2 + j -1
                    index = index.astype(int)
                    f_partial[index,0] += forza_nodo
                    
                    #print(f_partial[index,0])

    return f_partial

def f_assembly(P, T, Results_dir, num_procs=4):
    """Parallelizza il calcolo di K_global usando multiprocessing"""
    n_elementi = T.shape[1]
    chunk_size = n_elementi // num_procs
    ranges = [range(i, min(i + chunk_size, n_elementi)) for i in range(0, n_elementi, chunk_size)]

    with Pool(processes=num_procs) as pool:
        f_partials = pool.map(f_assemble_partial, [(P, T, r, Results_dir) for r in ranges])

    # Sommare tutte le matrici sparse parziali in una globale
    f_global = sum(f_partials, lil_matrix((3 * P.shape[1],1)))

    f_global = f_global.reshape((1,3 * P.shape[1]))
    
    return f_global


def vincolo_x(K, f, P, input_dir):
    n_nodi = P.shape[1]
    nodi_vincolati = []

    # Costruisci il percorso relativo al file
    Dirichlet_BC_path = os.path.join(input_dir, 'Dirichlet_BC.txt')

    Dirichlet_BC = np.loadtxt(Dirichlet_BC_path, delimiter=',')

    nodi_vincolati_out = Dirichlet_BC[:,0].astype(int)
    cond_BC_out = np.array([Dirichlet_BC[:,1],Dirichlet_BC[:,2],Dirichlet_BC[:,3]]).T
    def_BC_out = np.array([Dirichlet_BC[:,4],Dirichlet_BC[:,5],Dirichlet_BC[:,6]]).T
    
    # riordino gli indici in modo che siano in ordine decrescente
    indici_ordinati = np.argsort(Dirichlet_BC[:, 0])[::-1]
    Dirichlet_BC = Dirichlet_BC[indici_ordinati,:]

    nodi_vincolati = Dirichlet_BC[:,0].astype(int)
    cond_BC = np.array([Dirichlet_BC[:,1],Dirichlet_BC[:,2],Dirichlet_BC[:,3]]).T
    def_BC = np.array([Dirichlet_BC[:,4],Dirichlet_BC[:,5],Dirichlet_BC[:,6]]).T
    
    idx_to_remove = np.array([], dtype=int)
    for i in range(0,np.size(nodi_vincolati,0), 1):

        for j in range(2,-1,-1):

            if cond_BC[i,j] == 1:
                axis = 2-j
                index = 3*nodi_vincolati[i] -axis -1 

                f_new = def_BC[i,j]*K[:,index]

                idx_to_remove = np.append(idx_to_remove,index)

                f = f-f_new.T

    idx_to_keep = [i for i in range(K.shape[0]) if i not in idx_to_remove]
    K_new = K[idx_to_keep, :][:, idx_to_keep]           
    f_new = np.delete(f,idx_to_remove,1)

    return K_new, f_new.T, nodi_vincolati_out, cond_BC_out, def_BC_out


from scipy.sparse.linalg import cg
from scipy.sparse import diags
from scipy.sparse import csr_matrix

def lin_syst_solver_par(K, f, toll=1e-6):
    
    # Calcola il precondizionatore diagonale
    M_diag = diags(1.0 / K.diagonal())

    print('Precondizionatore calcolato.')

    # Risolvi il sistema lineare utilizzando il metodo del gradiente coniugato
    u_new, info = cg(K, f, x0=None, rtol=toll, atol=0.0, maxiter=10000, M=M_diag, callback=None)
    
    if info != 0:
        print(f"Il metodo del gradiente coniugato non è riuscito a convergere. Codice di errore: {info}")
    
    return u_new


def u_composition(u_new, n_nodi, nodi_vincolati, cond_BC, def_BC):

    u = np.array([])
    index = 0
    for i in range(1,n_nodi+1):

        check = 1

        for j in range(len(nodi_vincolati)):

            if i==nodi_vincolati[j]:
                
                if cond_BC[j,0] == 1:
                    u = np.append(u,def_BC[j,0])
                else:
                    u = np.append(u,u_new[index])
                    index += 1
                
                if cond_BC[j,1] == 1:
                    u = np.append(u,def_BC[j,1])
                else:
                    u = np.append(u,u_new[index])
                    index += 1

                if cond_BC[j,2] == 1:
                    u = np.append(u,def_BC[j,2])
                else:
                    u = np.append(u,u_new[index])
                    index += 1

                check = 0
                break

        if check == 1 :
            u = np.append(u,np.array([u_new[index],u_new[index+1],u_new[index+2]]))
            index += 3
            
    return u


def von_mises(sigma):
    # sigma è un vettore 6x1: [sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_zx]
    sigma_xx = sigma[0]
    sigma_yy = sigma[1]
    sigma_zz = sigma[2]
    tau_xy = sigma[3]
    tau_yz = sigma[4]
    tau_zx = sigma[5]
    
    # Formula dello sforzo di Von Mises
    von_mises = np.sqrt(0.5 * ((sigma_xx - sigma_yy)**2 + (sigma_yy - sigma_zz)**2 + (sigma_zz - sigma_xx)**2 + 
                               6 * (tau_xy**2 + tau_yz**2 + tau_zx**2)))
    
    return von_mises


def sigma_assembly1(P, T, u, E, nu):
    n_nodi = P.shape[1]
    n_elementi = T.shape[1]

    sigma = np.zeros((6, n_elementi))
    stress_vm = np.zeros(n_elementi)
    stress_vm_nodes = np.zeros(n_nodi)
    counter = np.zeros(n_nodi)

    for i in range(n_elementi):
        nodi = T[:, i].astype(int)
        coord_nodi = P[:, nodi-1]

        u_e1 = np.array([u[nodi[0]*3-3], u[nodi[0]*3-2],u[nodi[0]*3-1]]).T
        u_e2 = np.array([u[nodi[1]*3-3], u[nodi[1]*3-2],u[nodi[1]*3-1]]).T
        u_e3 = np.array([u[nodi[2]*3-3], u[nodi[2]*3-2],u[nodi[2]*3-1]]).T
        u_e4 = np.array([u[nodi[3]*3-3], u[nodi[3]*3-2],u[nodi[3]*3-1]]).T
        u_element = np.concatenate((u_e1, u_e2, u_e3, u_e4))

        Dp = DN_assembly(coord_nodi)
        B = B_assembly(Dp)
        D = D_assembly(E, nu)

        eps = B @ u_element
        # sigma = (s_xx, s_yy, s_zz, t_xy, t_yz, t_xz)
        sigma_element = D @ eps
        sigma[:, i] = sigma_element

        stress_vm[i] = von_mises(sigma_element)

        for j in range(4):
            counter[nodi[j]-1] += 1
            stress_vm_nodes[nodi[j]-1] += stress_vm[i]

    # Calcolo stress_medio nei nodi
    for i in range(n_nodi):
        if counter[i] != 0:
            stress_vm_nodes[i] /= counter[i]

    return sigma, stress_vm, stress_vm_nodes


from multiprocessing import Pool
import numpy as np

def sigma_assembly_partial(args):
    """Calcola sigma, stress_vm e stress_vm_nodes per un sottoinsieme di elementi."""
    P, T, u, E, nu, elem_range = args
    n_nodi = P.shape[1]
    n_elementi = len(elem_range)

    sigma = np.zeros((6, n_elementi))
    stress_vm = np.zeros(n_elementi)
    stress_vm_nodes = np.zeros(n_nodi)
    counter = np.zeros(n_nodi)

    for idx, i in enumerate(elem_range):
        nodi = T[:, i].astype(int)
        coord_nodi = P[:, nodi - 1]

        u_e1 = np.array([u[nodi[0] * 3 - 3], u[nodi[0] * 3 - 2], u[nodi[0] * 3 - 1]]).T
        u_e2 = np.array([u[nodi[1] * 3 - 3], u[nodi[1] * 3 - 2], u[nodi[1] * 3 - 1]]).T
        u_e3 = np.array([u[nodi[2] * 3 - 3], u[nodi[2] * 3 - 2], u[nodi[2] * 3 - 1]]).T
        u_e4 = np.array([u[nodi[3] * 3 - 3], u[nodi[3] * 3 - 2], u[nodi[3] * 3 - 1]]).T
        u_element = np.concatenate((u_e1, u_e2, u_e3, u_e4))

        Dp = DN_assembly(coord_nodi)
        B = B_assembly(Dp)
        D = D_assembly(E, nu)

        eps = B @ u_element
        sigma_element = D @ eps
        sigma[:, idx] = sigma_element

        stress_vm[idx] = von_mises(sigma_element)

        for j in range(4):
            counter[nodi[j] - 1] += 1
            stress_vm_nodes[nodi[j] - 1] += stress_vm[idx]

    return sigma, stress_vm, stress_vm_nodes, counter


def sigma_assembly(P, T, u, E, nu, num_procs=4):
    """Parallelizza il calcolo di sigma_assembly."""
    n_nodi = P.shape[1]
    n_elementi = T.shape[1]

    # Suddividi gli elementi in intervalli per ogni processo
    chunk_size = n_elementi // num_procs
    ranges = [range(i, min(i + chunk_size, n_elementi)) for i in range(0, n_elementi, chunk_size)]

    # Prepara gli argomenti per ogni processo
    args = [(P, T, u, E, nu, r) for r in ranges]

    # Esegui i calcoli in parallelo
    with Pool(processes=num_procs) as pool:
        results = pool.map(sigma_assembly_partial, args)

    # Combina i risultati dai processi
    sigma = np.zeros((6, n_elementi))
    stress_vm = np.zeros(n_elementi)
    stress_vm_nodes = np.zeros(n_nodi)
    counter = np.zeros(n_nodi)

    for idx, (sigma_partial, stress_vm_partial, stress_vm_nodes_partial, counter_partial) in enumerate(results):
        start = idx * chunk_size
        end = start + len(stress_vm_partial)
        sigma[:, start:end] = sigma_partial
        stress_vm[start:end] = stress_vm_partial
        stress_vm_nodes += stress_vm_nodes_partial
        counter += counter_partial

    # Calcolo dello stress medio nei nodi
    for i in range(n_nodi):
        if counter[i] != 0:
            stress_vm_nodes[i] /= counter[i]

    return sigma, stress_vm, stress_vm_nodes
