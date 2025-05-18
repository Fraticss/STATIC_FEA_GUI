import pyvista as pv
import numpy as np
import os

def postprocess(Result_path, view_selection):

    # Ottieni il percorso della cartella in cui si trova lo script
    #Results_path = os.path.join(Result_path, 'Results')
    Output_folder = os.path.join(Result_path, 'Output')
    # Costruisci il percorso relativo ai file nella sottocartella 'Output'
    P_new_path = os.path.join(Output_folder, 'P_new.txt')
    T_path = os.path.join(Output_folder, 'mesh_T.txt')
    stress_vm_path = os.path.join(Output_folder, 'stress_vm.txt')
    stress_vm_nodi_path = os.path.join(Output_folder, 'stress_vm_nodi.txt')
    u_nodes_path = os.path.join(Output_folder, 'u_nodes.txt')

    # Carica i file
    P_new = np.loadtxt(P_new_path, delimiter=',')  # Trasponi per ottenere la matrice corretta
    T = np.loadtxt(T_path, delimiter=',').astype(int)
    stress_vm = np.loadtxt(stress_vm_path, delimiter=',')
    stress_vm_nodi = np.loadtxt(stress_vm_nodi_path, delimiter=',')
    u_nodes = np.loadtxt(u_nodes_path, delimiter=',')

    # Trasponi P_new → (n_nodi, 3)
    points = P_new.T

    n_elem = T.shape[1]

    print("Shape matrice T:", T.shape)
    print("Shape matrice P:", points.shape)

    cells = np.hstack([np.full((n_elem, 1), 4), T.T-1])  # ogni riga: [4, i1, i2, i3, i4]
    celltypes = np.full(n_elem, 10)  # Codice VTK per tetraedro

    # Crea la mesh tetraedrica
    grid = pv.UnstructuredGrid(cells, celltypes, points)
    
    if view_selection == 'Von Mises Stresses (Elements)':
        # Aggiungi campo scalare alla mesh negli elementi
        grid["Von Mises Stresses (Elements)"] = stress_vm
        bar = np.array([np.min(stress_vm),np.max(stress_vm)])
    elif view_selection == 'Von Mises Stresses (Nodes)':
        # Aggiungi campo scalare alla mesh negli elementi
        grid["Von Mises Stresses (Nodes)"] = stress_vm_nodi
        bar = np.array([np.min(stress_vm_nodi),np.max(stress_vm_nodi)])
    elif view_selection == 'Disp_x':
        # Aggiungi campo scalare alla mesh negli elementi
        grid["Disp_x"] = u_nodes[0::3]
        bar = np.array([np.min(u_nodes[0::3]),np.max(u_nodes[0::3])])
    elif view_selection == 'Disp_y':
        # Aggiungi campo scalare alla mesh negli elementi
        grid["Disp_y"] = u_nodes[1::3]
        bar = np.array([np.min(u_nodes[1::3]),np.max(u_nodes[1::3])])
    elif view_selection == 'Disp_z':
        # Aggiungi campo scalare alla mesh negli elementi
        grid["Disp_z"] = u_nodes[2::3]
        bar = np.array([np.min(u_nodes[2::3]),np.max(u_nodes[2::3])])

    '''
    # Aggiungi campo scalare alla mesh nei nodi
    grid["stress_magnitude"] = stress_vm_nodi
    '''
    # Visualizza colorando in base al modulo
    plotter = pv.Plotter()
    plotter.add_mesh(
        grid, 
        scalars=view_selection,
        cmap="jet", 
        show_edges=False, 
        clim=bar  # Imposta i limiti della scala di colori
    )

    # Aggiunge un sistema di riferimento in basso a sinistra
    plotter.add_axes()  

    plotter.show()  # Mostra la mesh