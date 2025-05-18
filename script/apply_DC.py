import numpy as np
import gmsh
import sys
import os


def DC_BC(Out_path):
    # Costruisci il percorso del file .msh nella sottocartella Input
    Results_path = os.path.join(Out_path, 'Results')
    input_dir = os.path.join(Results_path, 'Input')
    mesh_file = os.path.join(input_dir, 'MESH_FILE_4x.msh')  # Sostituisci con il nome del file .msh
    data_path = os.path.join(input_dir, 'DC_constraints.txt')  # Sostituisci con il nome del file .msh

    # Inizializza Gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)  # Abilita output nel terminale per debug
    gmsh.open(mesh_file)

    print(f"Mesh caricata da: {mesh_file}")

    # Definizione delle condizioni al contorno di Dirichlet
    data = np.loadtxt(data_path, delimiter=',')  # Carica le condizioni al contorno di Dirichlet

    # Assicurati che data sia una matrice con due dimensioni
    data = np.atleast_2d(data)

    # Se data è un vettore riga, trasponilo per ottenere una matrice (n, 1)
    if data.shape[0] == 1:
        data = data.T

    face_DC = np.array(data[0,:], dtype=int) 
    DC_BC = np.array(data[1:4,:])  # Ogni colonna fa riferimento a una faccia e ogni riga alla coordinata del nodo (0 = non vincolato, 1 = vincolato)
    DC_def = np.array(data[4:,:])  # Spostamenti imposti lungo x, y, z per ogni faccia

    # Inizializza gli array vuoti
    D_nodes = []
    D_BC_x = []
    D_BC_y = []
    D_BC_z = []
    D_coord_x = []
    D_coord_y = []
    D_coord_z = []

    # Itera sulle facce specificate
    for i in range(len(face_DC)):
        # Trova i nodi appartenenti alla faccia usando Gmsh
        node_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(2, face_DC[i])

        # Check if nodes were found
        if len(node_tags) == 0:
            print(f"Errore: Nessun nodo trovato per il gruppo fisico con indice {face_DC[i]}.")
            continue

        # Aggiungi i nodi e le condizioni al contorno
        D_nodes.extend(node_tags)
        D_BC_x.extend([DC_BC[0, i]] * len(node_tags))
        D_BC_y.extend([DC_BC[1, i]] * len(node_tags))
        D_BC_z.extend([DC_BC[2, i]] * len(node_tags))
        D_coord_x.extend([DC_def[0, i]] * len(node_tags))
        D_coord_y.extend([DC_def[1, i]] * len(node_tags))
        D_coord_z.extend([DC_def[2, i]] * len(node_tags))

    # Combina i dati in una matrice Dirichlet_BC
    Dirichlet_BC = np.column_stack([D_nodes, D_BC_x, D_BC_y, D_BC_z, D_coord_x, D_coord_y, D_coord_z])

    # Costruisci il percorso del file di output per le condizioni al contorno di Dirichlet
    Dirichlet_BC_path = os.path.join(input_dir, 'Dirichlet_BC.txt')

    # Salva i dati di Dirichlet_BC in un file
    np.savetxt(Dirichlet_BC_path, Dirichlet_BC, fmt='%d', delimiter=',')
    print(f"Condizioni al contorno di Dirichlet salvate in: {Dirichlet_BC_path}")

    # Finalize Gmsh
    gmsh.finalize()
