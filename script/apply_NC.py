import numpy as np
import gmsh
import sys
import os


def NC_BC(Out_path):

    # Costruisci il percorso del file .msh nella sottocartella Input
    Results_path = os.path.join(Out_path, 'Results')
    input_dir = os.path.join(Results_path, 'Input')
    mesh_file = os.path.join(input_dir, 'MESH_FILE_4x.msh')  # Sostituisci con il nome del file .msh
    data_path = os.path.join(input_dir, 'NC_constraints.txt')  # Sostituisci con il nome del file .msh

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

    face_N = np.array(data[0,:], dtype=int) 
    N_BC = np.array(data[1:4,:])  # Ogni colonna fa riferimento a una faccia e ogni riga alla coordinata del nodo (0 = non vincolato, 1 = vincolato)
    N_stress = np.array(data[4:,:])  # Stress imposti lungo x, y, z per ogni faccia

    # Initialize arrays for Neumann BC
    N_nodes = []
    N_BC_x = []
    N_BC_y = []
    N_BC_z = []
    N_coord_x = []
    N_coord_y = []
    N_coord_z = []

    # Loop through the specified faces
    for i in range(len(face_N)):
        # Finds the nodes belonging to the face using Gmsh
        node_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(2, face_N[i])

        # Check if nodes were found
        if len(node_tags) == 0:
            print(f"Errore: Nessun nodo trovato per il gruppo fisico con indice {face_N[i]}.")
            continue

        # Append data for Neumann BC
        N_nodes.extend(node_tags)
        N_BC_x.extend([N_BC[0, i]] * len(node_tags))
        N_BC_y.extend([N_BC[1, i]] * len(node_tags))
        N_BC_z.extend([N_BC[2, i]] * len(node_tags))
        N_coord_x.extend([N_stress[0, i]] * len(node_tags))
        N_coord_y.extend([N_stress[1, i]] * len(node_tags))
        N_coord_z.extend([N_stress[2, i]] * len(node_tags))

    # Combine Neumann BC data into a single array
    Neumann_BC = np.column_stack([N_nodes, N_BC_x, N_BC_y, N_BC_z, N_coord_x, N_coord_y, N_coord_z])

    # Costruisci il percorso del file di output per le condizioni al contorno di Neumann
    Neumann_BC_path = os.path.join(input_dir, 'Neumann_BC.txt')

    # Salva i dati di Dirichlet_BC in un file
    np.savetxt(Neumann_BC_path, Neumann_BC, fmt='%d', delimiter=',')

    print(f"Neumann BC saved to: {Neumann_BC_path}")

    # Finalize Gmsh
    gmsh.finalize()

if __name__ == "__main__":
    # Esegui la funzione NC_BC con il percorso di input desiderato
    NC_BC(sys.argv[1])
