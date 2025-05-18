import pyvista as pv
import numpy as np
import os
import sys

def mesh_viewer(output_path):

    Results_path = os.path.join(output_path, 'Results')
    input_path = os.path.join(Results_path, 'Input')
    P_path = os.path.join(input_path, 'mesh_P.txt')
    T_path = os.path.join(input_path, 'mesh_T.txt')
    # Converti P e T in array numpy
    P = np.loadtxt(P_path, delimiter=",")  # Carica la matrice P
    T = np.loadtxt(T_path, delimiter=",", dtype=int)  # Carica la matrice T

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
    plotter.show(title="Mesh Viewer")
