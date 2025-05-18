import gmsh
import numpy as np
import sys
import os
import threading

def gen_mesh(model_path, output_path, h_max, h_min, print_func):

    print_func("...")
    # Inizializza gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0) # Disabilita l'output della console di gmsh

    # Carica il file STEP

    #step_file = os.path.join(script_dir, 'Input',"Cantilivier_beam.STEP")  # Cambia con il tuo file STEP
    gmsh.model.occ.importShapes(model_path)
    gmsh.model.occ.synchronize()

    # Definisci la dimensione massima degli elementi della mesh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h_min)  # Dimensione minima degli elementi
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h_max)  # Dimensione massima degli elementi

    # Imposta controlli sulla qualità della mesh
    gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.5)  # Soglia di qualità
    gmsh.option.setNumber("Mesh.Optimize", 1)  # Abilita ottimizzazione
    gmsh.option.setNumber("Mesh.QualityType", 2)  # Criterio di qualità: rapporto di aspetto

    # Genera la mesh 3D (tetraedrica)
    gmsh.model.mesh.generate(3)

    surfaces = gmsh.model.getEntities(2)  # Ottieni le superfici
    for  i,surface in enumerate(surfaces):
        #gmsh.model.setPhysicalName(2, surface[1], f"Surface_{i + 1}")
        gmsh.model.addPhysicalGroup(2, [surface[1]], tag=surface[1])  # Crea un gruppo fisico con il numero della faccia
        gmsh.model.setPhysicalName(2, surface[1], str(surface[1]))  # Usa solo il numero della faccia come nome

    print_func("...")
    # Salva la mesh in formato 4

    # Imposta il formato del file .msh a 4
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.SaveAll", 1)  # Salva tutto

    # Salva la mesh
    Results_path = os.path.join(output_path, 'Results')
    input_path = os.path.join(Results_path, 'Input')
    mesh4_file = os.path.join(input_path, 'MESH_FILE_4x.msh')
    gmsh.write(mesh4_file)  # Salva il file in formato MSH 4.1

    element_types, _, _ = gmsh.model.mesh.getElements()
    #print_func("Tipi di elementi generati:", element_types)

    print_func("...")
    # Salva la mesh in formato 2.2

    # Imposta il formato del file .msh a 2.2
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.SaveAll", 1)  # Salva tutto

    # Salva la mesh
    Results_path = os.path.join(output_path, 'Results')
    input_path = os.path.join(Results_path, 'Input')
    mesh22_file = os.path.join(input_path, 'MESH_FILE_22.msh')
    gmsh.write(mesh22_file)  # Salva il file in formato MSH 2.2

    print_func("... Mesh generated.")

    gmsh.finalize()  # Chiudi gmsh

    import meshio

    # Carica il file .msh
    mesh = meshio.read(mesh22_file)

    # Estrai la connettività degli elementi tetraedrici
    tetra = mesh.cells_dict.get("tetra")

    if tetra is None:
        raise ValueError("Il file non contiene elementi tetraedrici.")

    # 'tetra' è la matrice di connettività (n_elem, 4)
    T = tetra+1  # Aggiungi 1 per passare da indici 0-based a 1-based

    # Per completezza, estrai anche le coordinate dei nodi
    P = mesh.points

    #threading.Thread(target=print_func("Shape matrice T:", T.shape)).start()
    #threading.Thread(target=print_func("Shape matrice P:", P.shape)).start()
    print_func("Shape matrice T:", T.shape)
    print_func("Shape matrice P:", P.shape)

    # save matricies
    Results_path = os.path.join(output_path, 'Results')
    input_path = os.path.join(Results_path, 'Input')
    P_path = os.path.join(input_path, 'mesh_P.txt')
    T_path = os.path.join(input_path, 'mesh_T.txt')

    # Salva i file di output nella cartella Output
    np.savetxt(T_path, T, delimiter=',')
    np.savetxt(P_path, P, delimiter=',')

    print_func("Mesh saved.")

    import pyvista as pv

    # Estrai le coordinate dei nodi
    points = P[:, :]  # Colonne X, Y, Z

    # Estrai la connettività degli elementi
    connectivity = T[:, :]-1  # Colonne Node1, Node2, Node3, Node4 (PyVista usa indici 0-based)

    # Crea un array per la connettività in formato PyVista
    cells = np.hstack([np.full((connectivity.shape[0], 1), 4), connectivity]).flatten()

    # Definisci il tipo di elemento (10 = tetraedro in VTK)
    cell_type = np.full(connectivity.shape[0], 10)

    # Crea un oggetto UnstructuredGrid
    grid = pv.UnstructuredGrid(cells, cell_type, points)

    # Plotta la mesh
    plotter = pv.Plotter()
    plotter.add_mesh(grid, show_edges=True, color="lightblue", opacity=1.0)
    plotter.show()
