import gmsh
import numpy as np
import sys
import os

def view_step(path):
    
    # Inizializza gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)

    #step_file = os.path.join(script_dir, 'Input',"Cantilivier_beam.STEP")  # Cambia con il tuo file STEP
    gmsh.model.occ.importShapes(path)  # Importa il file STEP
    gmsh.model.occ.synchronize()

    # Genera la mesh 3D (tetraedrica)
    gmsh.model.mesh.generate(3)

    surfaces = gmsh.model.getEntities(2)  # Ottieni le superfici
    for  i,surface in enumerate(surfaces):
        #gmsh.model.setPhysicalName(2, surface[1], f"Surface_{i + 1}")
        gmsh.model.addPhysicalGroup(2, [surface[1]], tag=surface[1])  # Crea un gruppo fisico con il numero della faccia
        gmsh.model.setPhysicalName(2, surface[1], str(surface[1]))  # Usa solo il numero della faccia come nome

    # Avvia l'interfaccia grafica per visualizzare la mesh
    gmsh.option.setNumber("Geometry.PointNumbers", 0)
    gmsh.option.setNumber("Geometry.SurfaceNumbers", 1)
    gmsh.option.setNumber("Geometry.LabelType", 2)
    gmsh.option.setNumber("Mesh.SurfaceEdges", 0)
    gmsh.option.setNumber("Mesh.SurfaceFaces", 0)
    gmsh.option.setNumber("Mesh.VolumeEdges", 0)
    gmsh.option.setNumber("Mesh.VolumeFaces", 0)

    # Set GUI background to black and text to white
    gmsh.option.setNumber("General.ColorScheme", 3)  # Set dark mode
    gmsh.option.setNumber("General.Light0", 0)  # Turn off light 0

    # Avvia l'interfaccia grafica per visualizzare la mesh
    gmsh.fltk.run()
    
    # Imposta modalità di visualizzazione scura


    # Finalizza gmsh
    gmsh.finalize()

