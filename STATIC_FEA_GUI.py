import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys
import numpy as np
import multiprocessing
import tkinter.font as tkFont
import importlib

class FEMApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("STATIC FEA GUI")
        #self.geometry("700x500")

        # Ottieni il percorso corretto in base a se il programma è in esecuzione come .py o .exe
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(__file__)

        icon_path = os.path.join(base_path, "FEA_logo.ico")

        self.iconbitmap(icon_path)

        self.file_step = None

        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)

        self.steps = []
        # Qui passiamo self come controller
        for Step in (Step1, Step2, Step3, Step4, Step5, Step6):
            page = Step(self.container, self)  # Passa 'self' come controller
            self.steps.append(page)
            page.grid(row=0, column=0, sticky="nsew")

        self.current_step = 0  # Traccia lo step corrente
        self.show_step(0)

    def show_step(self, index):
        self.current_step = index
        frame = self.steps[index]
        frame.tkraise()

        # Mostra il messaggio di avviso solo per Step2
        if isinstance(frame, Step2):
            frame.show_notice_step2()
        elif isinstance(frame, Step5):
            frame.show_notice_step5()

    def previous_step(self):
        if self.current_step > 0:
            self.show_step(self.current_step - 1)

class Step1(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Colonna sinistra
        left_frame = tk.Frame(self)
        left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # Sottocolonna 1: Elenco degli step
        steps_frame = tk.Frame(left_frame, bg="lightgray")
        steps_frame.pack(side="left", fill="both", padx=0, pady=0)

        # Configura il font di default
        default_font = tkFont.nametofont("TkDefaultFont")
        default_font.configure(size=14, weight="bold")

        # Elenco degli step
        self.steps_titles = [
            "Step 1: Select .STEP file",
            "Step 2: Mesh Generation",
            "Step 3: Position Constraints",
            "Step 4: Pressure Constraints",
            "Step 5: Solver",
            "Step 6: Results Visualization"
        ]
        self.step_labels = []
        for i, title in enumerate(self.steps_titles):
            label = tk.Label(
                steps_frame,
                text=title,
                bg="lightgray",
                fg="black",
                anchor="w",
                padx=10
            )
            label.pack(fill="x", pady=5)
            self.step_labels.append(label)

        # Evidenzia lo step corrente
        self.highlight_current_step(0)

        # Sottocolonna 2: Pulsanti principali
        buttons_frame = tk.Frame(left_frame)
        buttons_frame.pack(side="left", fill="both", expand=True, padx=0, pady=0)

        tk.Label(buttons_frame, text="Step 1: Select .STEP file", font=("Arial", 14)).pack(pady=10)

        tk.Button(buttons_frame, text="Choose File", command=self.choose_file).pack(pady=5)

        tk.Button(buttons_frame, text="Select Output Folder", command=self.select_output_folder).pack(pady=10)

        tk.Button(buttons_frame, text="View Model", command=self.view_model).pack(pady=10)
        tk.Button(buttons_frame, text="Next", command=self.next_step).pack(pady=10)

        # Colonna destra: Percorsi delle cartelle selezionate
        right_frame = tk.Frame(self, bg="white", width = 300)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.path_label = tk.Label(right_frame, text="No file selected", fg="gray", wraplength=300)
        self.path_label.pack(pady=5)

        self.output_folder_label = tk.Label(right_frame, text="No folder selected", fg="gray", wraplength=300)
        self.output_folder_label.pack(pady=5)

        # Aggiungi una sezione di testo esplicativo nella parte bassa
        explanation_label = tk.Label(
            buttons_frame,
            text="EXPLANATION:\n"
            "1) CHOOSE FILE: you need to select a .STEP file that represents the geometry of your model.\n"
            "2) SELECT OUTPUT FOLDER: you need to select a folder where the results will be saved.\n"
            "3) VIEW MODEL: you can visualize the selected model in GMSH.\n"
            "4) NEXT: proceed to the next step.",
            font=("Arial", 14),
            wraplength=400,
            justify="left",
            fg="black",
            bg="#FFFACD"  # Giallo chiaro
        )
        explanation_label.pack(pady=20, anchor="s")  # Posiziona il testo nella parte bassa

    def highlight_current_step(self, current_step):
        """Evidenzia lo step corrente nell'elenco degli step."""
        for i, label in enumerate(self.step_labels):
            if i == current_step:
                label.config(bg="yellow", font=("Arial", 14, "bold"))
            else:
                label.config(bg="lightgray", font=("Arial", 14))

    def choose_file(self):
        path = filedialog.askopenfilename(filetypes=[("STEP Files", "*.step *.stp")])
        if path:
            self.controller.file_step = path
            self.path_label.config(text=f"Selected file:\n{path}", fg="black")

    def select_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.controller.output_folder = folder
            self.output_folder_label.config(text=f"Selected folder:\n{folder}", fg="black")

    def view_model(self):
        if not self.controller.file_step:
            messagebox.showwarning("Warning", "Please select a STEP file first.")
            return

        try:
            # Import dinamico dello script
            module = importlib.import_module("script.view_model")
            module.view_step(self.controller.file_step)
        except Exception as e:
            messagebox.showerror("Errore", f"Errore durante la visualizzazione: {e}")

    def next_step(self):
        if not self.controller.file_step:
            messagebox.showwarning("Warning", "Please select a STEP file first.")
            return
        if not hasattr(self.controller, 'output_folder') or not self.controller.output_folder:
            messagebox.showwarning("Warning", "Please select an output folder first.")
            return

        # Create the Results folder in the selected output folder
        results_folder = os.path.join(self.controller.output_folder, "Results")
        try:
            os.makedirs(results_folder, exist_ok=True)
            print(f"Results folder created at: {results_folder}")  # Debug message
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create Results folder: {e}")
            return
        
        Results_path = os.path.join(self.controller.output_folder, "Results")
        Input_folder = os.path.join(Results_path, "Input")
        try:
            os.makedirs(Input_folder, exist_ok=True)
            print(f"Results folder created at: {Input_folder}")  # Debug message
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create Results folder: {e}")
            return
        
        Output_folder = os.path.join(Results_path, "Output")
        try:
            os.makedirs(Output_folder, exist_ok=True)
            print(f"Results folder created at: {Output_folder}")  # Debug message
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create Results folder: {e}")
            return

        # Proceed to the next step
        self.controller.show_step(1)

class Step2(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        # Colonna sinistra
        left_frame = tk.Frame(self)
        left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # Sottocolonna 1: Elenco degli step
        steps_frame = tk.Frame(left_frame, bg="lightgray", width=200)
        steps_frame.pack(side="left", fill="y", padx=10, pady=0)

        # Sottocolonna 2: Pulsanti principali e output mesher
        main_frame = tk.Frame(left_frame, bg=None)
        main_frame.pack(side="left", fill="both", expand=True, padx=(10,0), pady=0)

        # Right column
        right_frame = tk.Frame(self, bg="white")
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Elenco degli step
        self.steps_titles = [
            "Step 1: Select .STEP file",
            "Step 2: Mesh Generation",
            "Step 3: Position Constraints",
            "Step 4: Pressure Constraints",
            "Step 5: Solver",
            "Step 6: Results Visualization"
        ]
        self.step_labels = []
        for i, title in enumerate(self.steps_titles):
            label = tk.Label(
                steps_frame,
                text=title,
                font=("Arial", 12),
                bg="lightgray",
                fg = "black",
                anchor="w",
                padx=10
            )
            label.pack(fill="x", pady=5)
            self.step_labels.append(label)

        # Evidenzia lo step corrente
        self.highlight_current_step(1)

        # Title label for Step 2
        tk.Label(main_frame, text="Step 2: Mesh Generation", font=("Arial", 14)).pack(pady=10)

        # Centering right frame content
        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_columnconfigure(2, weight=1)

        # Aggiungi i widget al centro
        content_frame = tk.Frame(right_frame, bg="white")
        content_frame.grid(row=1, column=1)

        # Input for h_max
        tk.Label(content_frame, text="h_max =", font=("Arial", 12)).grid(row=0, column=0, padx = 5, pady=5)
        self.h_max_entry = tk.Entry(content_frame, width=10)
        self.h_max_entry.grid(row=0, column=1, padx=5, pady=5)

        # Input for h_min
        tk.Label(content_frame, text="h_min =", font=("Arial", 12)).grid(row=1, column=0, padx = 5, pady=5)
        self.h_min_entry = tk.Entry(content_frame, width=10)
        self.h_min_entry.grid(row=1, column=1, padx=5, pady=5)

        # Button to generate mesh
        tk.Button(content_frame, text="Generate Mesh", command=self.generate_mesh).grid(row=2, column=0, columnspan=2, pady=10)

        # Buttons on the right
        tk.Button(main_frame, text="View Current Mesh", command=self.view_last_mesh).pack(pady=10)
        tk.Button(main_frame, text="Next", command=self.next_step_mesh).pack(pady=10)
        tk.Button(main_frame, text="Back", command=controller.previous_step).pack(pady=10)

        # Terminal output display on the right
        tk.Label(main_frame, text="Mesh Generation Output:", font=("Arial", 12)).pack(anchor="w", pady=5)
        self.output_text = tk.Text(main_frame, wrap="word", state="disabled", height=20)
        self.output_text.pack(fill="both", expand=True)

        # Aggiungi una sezione di testo esplicativo nella parte bassa
        explanation_label = tk.Label(
            main_frame,
            text="EXPLANATION:\n" 
            "1) h_max: select the value of the maximum length of every thetradrical element.\n"
            "2) h_min = select the minimum length of every thetraedrical element.\n" 
            "3) GENERATE MESH: you can generate a tethraedrical mesh in GMSH with the specifics given before. After the computation, a window with the mesh will be opened.\n" 
            "4) VIEW CURRENT MESH: visualize the last 3D mesh generated. If it is not as you espected, the parameters h can be changed and mesh can be generated again.\n" 
            "5)NEXT: proceed to the next step.\n" 
            "6) BACK: go back to the previous step.",
            font=("Arial", 12),
            wraplength=400,
            justify="left",
            fg="black",
            bg="#FFFACD"  # Giallo chiaro
        )
        explanation_label.pack(pady=20, anchor="s")  # Posiziona il testo nella parte bassa


        self.controller.mesh_gen_counter = 0

    def highlight_current_step(self, current_step):
        """Evidenzia lo step corrente nell'elenco degli step."""
        for i, label in enumerate(self.step_labels):
            if i == current_step:
                label.config(bg="yellow", font=("Arial", 10, "bold"))
            else:
                label.config(bg="lightgray", font=("Arial", 10))

    def show_notice_step2(self):
        """Mostra il messaggio di avviso quando lo Step2 viene visualizzato."""
        messagebox.showinfo(
            "Notice",
            "The h_max value should be selected based on the maximum size of the 3D model. "
            "If unsure, check the CAD model and start with high values of h_max, then regenerate with lower values if needed. "
            "Set h_min to 0 if there are no specific requirements."
        )

    def generate_mesh(self):
        # Get h_max and h_min values
        h_max = self.h_max_entry.get()
        h_min = self.h_min_entry.get()

        if not h_max or not h_min:
            messagebox.showwarning("Warning", "Please fill in both h_max and h_min.")
            return

        try:
            h_max = float(h_max)
            h_min = float(h_min)
        except ValueError:
            messagebox.showerror("Error", "h_max and h_min must be numbers.")
            return
        
        # Save h_max and h_min in the controller
        self.controller.h_max = h_max
        self.controller.h_min = h_min
        
        # Clear the output text widget
        self.output_text.config(state="normal")
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Generating mesh...\n")
        self.output_text.config(state="disabled")

        # Callback per scrivere nel widget
        def print_to_output(*args, sep=" ", end="\n"):
            text = sep.join(str(arg) for arg in args) + end
            self.output_text.config(state="normal")
            self.output_text.insert(tk.END, text)
            self.output_text.see(tk.END)
            self.output_text.config(state="disabled")
            self.output_text.update_idletasks()  # FORZA l'aggiornamento immediato


        # Funzione da eseguire in un thread
        def run_script():
            try:
                module = importlib.import_module("script.generate_mesh")
                module.gen_mesh(
                    self.controller.file_step,
                    self.controller.output_folder,
                    h_max,
                    h_min,
                    print_func=print_to_output
                )
            except Exception as e:
                print_to_output(f"Errore durante la generazione mesh: {e}")

        run_script()  # Esegui la funzione sul thread principale
        self.controller.mesh_gen_counter = 1
    
    def view_last_mesh(self):
        if self.controller.mesh_gen_counter == 0:
            messagebox.showwarning("Warning", "Please, first generate the mesh.")
            return

        try:
            # Import dinamico dello script
            module = importlib.import_module("script.view_mesh")
            module.mesh_viewer( self.controller.output_folder)
        except Exception as e:
            messagebox.showerror("Error", f"Error during the mesh visualization: {e}")

    def next_step_mesh(self):
        if self.controller.mesh_gen_counter == 0:
            messagebox.showwarning("Warning", "Please generate the mesh.")
            return

        # Proceed to the next step
        self.controller.show_step(2)

class Step3(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Left Frame
        left_frame = tk.Frame(self, bg=None)
        left_frame.pack(side="left", fill="both", expand=False, padx=10, pady=10)

        steps_frame = tk.Frame(left_frame, bg="lightgray", width=200)
        steps_frame.pack(side="left", fill="y", padx=0, pady=0)

        main_frame = tk.Frame(left_frame, bg=None)
        main_frame.pack(side="left", fill="both", expand=True, padx=(10,0), pady=0)

        # Right Frame
        right_frame = tk.Frame(self, bg = "white")
        right_frame.pack(side="right", fill="both", padx=10, pady=10, expand=True)

        # Title label for Step 3
        tk.Label(main_frame, text="Step 3: Position Constraints", font=("Arial", 14)).pack(pady=10)

        # Aggiungi le intestazioni sopra le colonne facce, x, y, z
        tk.Label(right_frame, text="Faces", font=("Arial", 12)).grid(row=1, column=0, pady=5)
        tk.Label(right_frame, text="DOF constrained (0 no, 1 yes)", font=("Arial", 12)).grid(row=1, column=1, pady=5)
        tk.Label(right_frame, text="x y z", font=("Arial", 12)).grid(row=1, column=2, pady=10)

        # Frame per le righe delle facce e dei vincoli
        self.face_entries = []  # Per tenere traccia dei campi delle facce
        self.spost_entries = []  # Per tenere traccia dei campi di spostamento
        self.constraint_entries = []  # Per tenere traccia dei vincoli (x, y, z)
        self.constraint_frames = []  # Per tenere traccia dei frame dei vincoli
        self.spost_frames = []

        # Variabile per tenere traccia della riga corrente
        self.current_row = 2  # Iniziamo dalla riga 2, poiché la riga 1 è occupata dalle intestazioni

        # Aggiungi il primo campo per il numero della faccia
        self.add_face_entry(right_frame)

        # Pulsanti per aggiungere e rimuovere righe
        button_frame = tk.Frame(right_frame)
        button_frame.grid(row=self.current_row, column=3, padx=10)

        add_button = tk.Button(button_frame, text="+", command=lambda: self.add_face_entry(right_frame))
        add_button.pack(side="left", padx=5)

        remove_button = tk.Button(button_frame, text="-", command=lambda: self.remove_face_entry(right_frame))
        remove_button.pack(side="left", padx=5)

        # Bottoni per la visualizzazione, il salvataggio e il passaggio al prossimo step
        tk.Button(main_frame, text="View Model", command=self.view_model).pack(pady=10, padx=5)
        tk.Button(main_frame, text="Next", command=self.next_step_DC).pack(pady=10)
        tk.Button(main_frame, text="Back", command=controller.previous_step).pack(pady=10)

        # Aggiungi una sezione di testo esplicativo nella parte bassa
        explanation_label = tk.Label(
            main_frame,
            text="EXPLANATION:\n"
            "1) VIEW MODEL: you can visualize the selected model in GMSH, to understand which faces you want to constrain.\n"
            "2) FACES: faces on which you want to apply a constraint on displacement.\n"
            "3) DOF: Directions (x,y,z) along you want your constraints. 1 means direction constrained, 0 means no constraint.\n"
            "4) X Y Z: on every cell you have to write the effective displacement that you want to impose along every direction (0 if you want that the faces is fixed along that direction).\n" 
            "5) +-: this two buttons allow you to add another face to constrain or to remove the last one added. (At least one face must be constrained since the structure needs to be at least isostatically constrained to have a meaningfull result from FEM.).\n" 
            "6) NEXT: proceed to the next step.\n" 
            "7) BACK: go back to the previous step.",
            font=("Arial", 12),
            wraplength=400,
            justify="left",
            fg="black",
            bg="#FFFACD"  # Giallo chiaro
        )
        explanation_label.pack(pady=20, anchor="s")  # Posiziona il testo nella parte bassa
        
        # Elenco degli step
        self.steps_titles = [
            "Step 1: Select .STEP file",
            "Step 2: Mesh Generation",
            "Step 3: Position Constraints",
            "Step 4: Pressure Constraints",
            "Step 5: Solver",
            "Step 6: Results Visualization"
        ]
        self.step_labels = []
        for i, title in enumerate(self.steps_titles):
            label = tk.Label(
                steps_frame,
                text=title,
                font=("Arial", 12),
                bg="lightgray",
                fg="black",
                anchor="w",
                padx=10
            )
            label.pack(fill="x", pady=5)
            self.step_labels.append(label)

        # Evidenzia lo step corrente
        self.highlight_current_step(2)

    def highlight_current_step(self, current_step):
        """Evidenzia lo step corrente nell'elenco degli step."""
        for i, label in enumerate(self.step_labels):
            if i == current_step:
                label.config(bg="yellow", font=("Arial", 10, "bold"))
            else:
                label.config(bg="lightgray", font=("Arial", 10))


    def add_face_entry(self, parent):
        """Aggiungi un nuovo campo per il numero della faccia, i vincoli DOF e i valori di spostamento"""
        # Crea una riga per una faccia
        face_entry = tk.Entry(parent)
        face_entry.grid(row=self.current_row, column=0, pady=5, padx=10)
        self.face_entries.append(face_entry)

        # Aggiungi i campi per i vincoli DOF (x, y, z) accanto
        constraint_frame = tk.Frame(parent)
        constraint_frame.grid(row=self.current_row, column=1, pady=5, padx=10)
        self.constraint_frames.append(constraint_frame)  # Salva il frame

        constraint_x = tk.Entry(constraint_frame, width=5)
        constraint_x.grid(row=0, column=0)
        constraint_y = tk.Entry(constraint_frame, width=5)
        constraint_y.grid(row=0, column=1)
        constraint_z = tk.Entry(constraint_frame, width=5)
        constraint_z.grid(row=0, column=2)

        self.constraint_entries.append((constraint_x, constraint_y, constraint_z))

        # Aggiungi i campi per i valori di spostamento (x, y, z) accanto
        spost_frame = tk.Frame(parent)
        spost_frame.grid(row=self.current_row, column=2, pady=5, padx=10)
        self.spost_frames.append(spost_frame)  # Salva il frame

        spost_x = tk.Entry(spost_frame, width=5)
        spost_x.grid(row=0, column=0)
        spost_y = tk.Entry(spost_frame, width=5)
        spost_y.grid(row=0, column=1)
        spost_z = tk.Entry(spost_frame, width=5)
        spost_z.grid(row=0, column=2)

        self.spost_entries.append((spost_x, spost_y, spost_z))

        # Incrementa la riga per la prossima faccia
        self.current_row += 1
    
    def remove_face_entry(self, parent):
        """Rimuovi l'ultima riga aggiunta, mantenendo almeno una riga"""
        if len(self.face_entries) > 1:
            # Rimuovi l'ultima entry per il numero della faccia
            face_entry = self.face_entries.pop()
            face_entry.destroy()

            # Rimuovi l'ultimo frame per i vincoli DOF
            constraint_frame = self.constraint_frames.pop()
            constraint_frame.destroy()

            # Rimuovi l'ultima entry per i vincoli DOF
            constraint_x, constraint_y, constraint_z = self.constraint_entries.pop()
            constraint_x.destroy()
            constraint_y.destroy()
            constraint_z.destroy()

            # Rimuovi l'ultimo frame per i valori di spostamento
            spost_frame = self.spost_frames.pop()
            spost_frame.destroy()

            # Rimuovi l'ultima entry per i valori di spostamento
            spost_x, spost_y, spost_z = self.spost_entries.pop()
            spost_x.destroy()
            spost_y.destroy()
            spost_z.destroy()

            # Decrementa la riga corrente
            self.current_row -= 1

    def view_model(self):
        if not self.controller.file_step:
            messagebox.showwarning("Warning", "Please select a STEP file first.")
            return

        try:
            # Import dinamico dello script
            module = importlib.import_module("script.view_model")
            module.view_step(self.controller.file_step)
        except Exception as e:
            messagebox.showerror("Errore", f"Errore durante la visualizzazione: {e}")

    def save_DC(self):
        # Recupera i numeri delle facce da vincolare
        DC_faces = [entry.get() for entry in self.face_entries if entry.get()]
        if not DC_faces:
            messagebox.showwarning("Warning", "Please enter at least one face number.")
            return

        # Recupera i vincoli per ciascuna faccia
        DC_constraints = []
        for constraint_x, constraint_y, constraint_z in self.constraint_entries:
            x = constraint_x.get()
            y = constraint_y.get()
            z = constraint_z.get()

            if not x or not y or not z:
                messagebox.showwarning("Warning", "Please fill in all constraints (x, y, z) for each face.")
                return

            DC_constraints.append((x, y, z))

        # Recupera i vincoli per ciascuna faccia
        DC_spost = []
        for spost_x, spost_y, spost_z in self.spost_entries:
            x = spost_x.get()
            y = spost_y.get()
            z = spost_z.get()

            if not x or not y or not z:
                messagebox.showwarning("Warning", "Please fill in all constraints DOF indication for each face.")
                return

            DC_spost.append((x, y, z))

        DC_const = np.vstack((np.array(DC_faces, dtype = int), np.array(DC_constraints, dtype = int).T, np.array(DC_spost, dtype=float).T,))

        Results_path = os.path.join(self.controller.output_folder, "Results")
        Input_folder = os.path.join(Results_path, "Input")
        DC_path = os.path.join(Input_folder, "DC_constraints.txt")
        np.savetxt(DC_path, DC_const, delimiter=',')

        # Run a script to compute the Dirichlet BC
        try:
            # Import dinamico dello script
            module = importlib.import_module("script.apply_DC")
            module.DC_BC(self.controller.output_folder)
        except Exception as e:
            messagebox.showerror("Errore", f"Errore durante la visualizzazione: {e}")

        # Proceed to the next step
        self.controller.show_step(3)

    def next_step_DC(self):
        # Salva i vincoli DC
        self.save_DC()

class Step4(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

       # Left Frame
        left_frame = tk.Frame(self, bg=None)
        left_frame.pack(side="left", fill="both", expand=False, padx=10, pady=10)

        steps_frame = tk.Frame(left_frame, bg="lightgray", width=200)
        steps_frame.pack(side="left", fill="y", padx=0, pady=0)

        main_frame = tk.Frame(left_frame, bg=None)
        main_frame.pack(side="left", fill="both", expand=True, padx=(10,0), pady=0)

         # Right Frame
        right_frame = tk.Frame(self, bg = "white")
        right_frame.pack(side="right", fill="both", padx=10, pady=10, expand=True)

        # Title label for Step 4
        tk.Label(main_frame, text="Step 4: Pressure Constraints", font=("Arial", 14)).pack(pady=10)

        # Aggiungi le intestazioni sopra le colonne facce, x, y, z
        tk.Label(right_frame, text="Faces", font=("Arial", 12)).grid(row=1, column=0, pady=5)
        tk.Label(right_frame, text="DOF constrained (0 no, 1 yes)", font=("Arial", 12)).grid(row=1, column=1, pady=5)
        tk.Label(right_frame, text="x y z", font=("Arial", 12)).grid(row=1, column=2, pady=10)

        # Frame per le righe delle facce e dei vincoli
        self.face_entries = []  # Per tenere traccia dei campi delle facce
        self.stress_entries = []  # Per tenere traccia dei campi di spostamento
        self.NBC_entries = []  # Per tenere traccia dei vincoli (x, y, z)
        self.NBC_frames = []  # Per tenere traccia dei frame dei vincoli
        self.stress_frames = []

        # Variabile per tenere traccia della riga corrente
        self.current_row = 2  # Iniziamo dalla riga 2, poiché la riga 1 è occupata dalle intestazioni

        # Aggiungi il primo campo per il numero della faccia
        self.add_face_entry(right_frame)

        # Pulsanti per aggiungere e rimuovere righe
        button_frame = tk.Frame(right_frame)
        button_frame.grid(row=self.current_row, column=3, padx=10)

        add_button = tk.Button(button_frame, text="+", command=lambda: self.add_face_entry(right_frame))
        add_button.pack(side="left", padx=5)

        remove_button = tk.Button(button_frame, text="-", command=lambda: self.remove_face_entry(right_frame))
        remove_button.pack(side="left", padx=5)

        # Bottoni per la visualizzazione, il salvataggio e il passaggio al prossimo step
        tk.Button(main_frame, text="View Model", command=self.view_model).pack(pady=10, padx=5)
        tk.Button(main_frame, text="Next", command=self.next_step_NC).pack(pady=10)
        tk.Button(main_frame, text="Back", command=controller.previous_step).pack(pady=10)

        # Aggiungi una sezione di testo esplicativo nella parte bassa
        explanation_label = tk.Label(
            main_frame,
            text="EXPLANATION:\n"
            "1) VIEW MODEL: you can visualize the selected model in GMSH, to understand which faces you want to constrain.\n"
            "2) FACES: faces on which you want to apply a pressure load on.\n"
            "3) DOF: Directions (x,y,z) along you want your constraints. 1 means direction constrained, 0 means no constraint.\n"
            "4) X Y Z: on every cell you have to write the effective pressure load that you want to impose along every direction (0 if you want that the faces is fixed along that direction).\n" 
            "5) +-: this two buttons allow you to add another face to constrain or to remove the last one added. (At least one face must be constrained since the structure needs to be at least isostatically constrained to have a meaningfull result from FEM.).\n" 
            "6) NEXT: proceed to the next step.\n" 
            "7) BACK: go back to the previous step.",
            font=("Arial", 12),
            wraplength=400,
            justify="left",
            fg="black",
            bg="#FFFACD"  # Giallo chiaro
        )
        explanation_label.pack(pady=20, anchor="s")  # Posiziona il testo nella parte bassa

        # Elenco degli step
        self.steps_titles = [
            "Step 1: Select .STEP file",
            "Step 2: Mesh Generation",
            "Step 3: Position Constraints",
            "Step 4: Pressure Constraints",
            "Step 5: Solver",
            "Step 6: Results Visualization"
        ]
        self.step_labels = []
        for i, title in enumerate(self.steps_titles):
            label = tk.Label(
                steps_frame,
                text=title,
                font=("Arial", 12),
                bg="lightgray",
                fg="black",
                anchor="w",
                padx=10
            )
            label.pack(fill="x", pady=5)
            self.step_labels.append(label)

        # Evidenzia lo step corrente
        self.highlight_current_step(3)

    def highlight_current_step(self, current_step):
        """Evidenzia lo step corrente nell'elenco degli step."""
        for i, label in enumerate(self.step_labels):
            if i == current_step:
                label.config(bg="yellow", font=("Arial", 10, "bold"))
            else:
                label.config(bg="lightgray", font=("Arial", 10))

    def add_face_entry(self, parent):
        """Aggiungi un nuovo campo per il numero della faccia, i vincoli DOF e i valori di spostamento"""
        # Crea una riga per una faccia
        face_entry = tk.Entry(parent)
        face_entry.grid(row=self.current_row, column=0, pady=5, padx=10)
        self.face_entries.append(face_entry)

        # Aggiungi i campi per i vincoli DOF (x, y, z) accanto
        NBC_frame = tk.Frame(parent)
        NBC_frame.grid(row=self.current_row, column=1, pady=5, padx=10)
        self.NBC_frames.append(NBC_frame)

        constraint_x = tk.Entry(NBC_frame, width=5)
        constraint_x.grid(row=0, column=0)
        constraint_y = tk.Entry(NBC_frame, width=5)
        constraint_y.grid(row=0, column=1)
        constraint_z = tk.Entry(NBC_frame, width=5)
        constraint_z.grid(row=0, column=2)

        self.NBC_entries.append((constraint_x, constraint_y, constraint_z))

        # Aggiungi i campi per i valori di spostamento (x, y, z) accanto
        stress_frame = tk.Frame(parent)
        stress_frame.grid(row=self.current_row, column=2, pady=5, padx=10)
        self.stress_frames.append(stress_frame)

        stress_x = tk.Entry(stress_frame, width=5)
        stress_x.grid(row=0, column=0)
        stress_y = tk.Entry(stress_frame, width=5)
        stress_y.grid(row=0, column=1)
        stress_z = tk.Entry(stress_frame, width=5)
        stress_z.grid(row=0, column=2)

        self.stress_entries.append((stress_x, stress_y, stress_z))

        # Incrementa la riga per la prossima faccia
        self.current_row += 1

    def remove_face_entry(self, parent):
        """Rimuovi l'ultima riga aggiunta, mantenendo almeno una riga"""
        if len(self.face_entries) > 1:
            # Rimuovi l'ultima entry per il numero della faccia
            face_entry = self.face_entries.pop()
            face_entry.destroy()

            # Rimuovi l'ultimo frame per i vincoli DOF
            NBC_frame = self.NBC_frames.pop()
            NBC_frame.destroy()

            # Rimuovi l'ultima entry per i vincoli DOF
            constraint_x, constraint_y, constraint_z = self.NBC_entries.pop()
            constraint_x.destroy()
            constraint_y.destroy()
            constraint_z.destroy()

            # Rimuovi l'ultimo frame per i valori di spostamento
            stress_frame = self.stress_frames.pop()
            stress_frame.destroy()

            # Rimuovi l'ultima entry per i valori di spostamento
            stress_x, stress_y, stress_z = self.stress_entries.pop()
            stress_x.destroy()
            stress_y.destroy()
            stress_z.destroy()

            # Decrementa la riga corrente
            self.current_row -= 1

    def view_model(self):
        if not self.controller.file_step:
            messagebox.showwarning("Warning", "Please select a STEP file first.")
            return

        try:
            # Import dinamico dello script
            module = importlib.import_module("script.view_model")
            module.view_step(self.controller.file_step)
        except Exception as e:
            messagebox.showerror("Errore", f"Errore durante la visualizzazione: {e}")

    def save_NC(self):
        # Recupera i numeri delle facce da vincolare
        NC_faces = [entry.get() for entry in self.face_entries if entry.get()]
        if not NC_faces:
            messagebox.showwarning("Warning", "Please enter at least one face number.")
            return

        # Recupera i vincoli per ciascuna faccia
        NC_constraints = []
        for constraint_x, constraint_y, constraint_z in self.NBC_entries:
            x = constraint_x.get()
            y = constraint_y.get()
            z = constraint_z.get()

            if not x or not y or not z:
                messagebox.showwarning("Warning", "Please fill in all constraints (x, y, z) for each face.")
                return

            NC_constraints.append((x, y, z))

        # Recupera i vincoli per ciascuna faccia
        NC_stress = []
        for stress_x, stress_y, stress_z in self.stress_entries:
            x = stress_x.get()
            y = stress_y.get()
            z = stress_z.get()

            if not x or not y or not z:
                messagebox.showwarning("Warning", "Please fill in all constraints DOF indication for each face.")
                return

            NC_stress.append((x, y, z))

        NC_const = np.vstack((np.array(NC_faces, dtype = int), np.array(NC_constraints, dtype = int).T, np.array(NC_stress, dtype=float).T))

        # Save the NC constraints to a file
        Results_path = os.path.join(self.controller.output_folder, "Results")
        Input_folder = os.path.join(Results_path, "Input")
        NC_path = os.path.join(Input_folder, "NC_constraints.txt")
        np.savetxt(NC_path, NC_const, delimiter=',')

        # Run a script to compute the Dirichlet BC
        try:
            # Import dinamico dello script
            module = importlib.import_module("script.apply_NC")
            module.NC_BC(self.controller.output_folder)
        except Exception as e:
            messagebox.showerror("Errore", f"Errore durante la visualizzazione: {e}")

        # Proceed to the next step
        self.controller.show_step(4)
    
    def next_step_NC(self):

        self.save_NC()

class Step5(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.controller.solver_counter = 0

        cores_available = multiprocessing.cpu_count()

        # Layout: Left for inputs, Right for terminal output
        left_frame = tk.Frame(self, bg="white")
        left_frame.pack(side="left", fill="both", expand=False, padx=20, pady=20)

        steps_frame = tk.Frame(left_frame, bg="lightgray", width=200)
        steps_frame.pack(side="left", fill="y", padx=0, pady=0)

        main_frame = tk.Frame(left_frame, bg=None)
        main_frame.pack(side="left", fill="both", expand=True, padx=(10,0), pady=0)

        right_frame = tk.Frame(self, bg="lightgray")
        right_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        # Input for E
        tk.Label(main_frame, text="E =", font=("Arial", 12)).grid(row=0, column=0, sticky="w", pady=5)
        self.e_entry = tk.Entry(main_frame, width=10)  # Slot largo almeno 5 cifre
        self.e_entry.grid(row=0, column=1, pady=5)

        # Input for nu
        tk.Label(main_frame, text="nu =", font=("Arial", 12)).grid(row=1, column=0, sticky="w", pady=5)
        self.nu_entry = tk.Entry(main_frame, width=5)  # Slot per almeno 3 cifre
        self.nu_entry.grid(row=1, column=1, pady=5)

        # Input for num_cores
        tk.Label(main_frame, text=f"num_cores (cores available = {cores_available}) =", font=("Arial", 12)).grid(row=2, column=0, sticky="w", pady=5)
        self.num_cores_entry = tk.Entry(main_frame, width=5)  # Slot per una cifra
        self.num_cores_entry.grid(row=2, column=1, pady=5)

        # Solve button
        tk.Button(main_frame, text="Solve", command=self.run_solver).grid(row=3, column=0, columnspan=2, pady=10)

        # Navigation buttons (in basso a sinistra)
        tk.Button(main_frame, text="Back", command=controller.previous_step).grid(row=4, column=0, pady=10, padx=10, sticky="w")
        tk.Button(main_frame, text="Next", command=self.next_step_solver).grid(row=4, column=1, pady=10, sticky="e")

        # Terminal output display
        tk.Label(right_frame, text="Solver Output:", font=("Arial", 12)).pack(anchor="w", pady=5)
        self.output_text = tk.Text(right_frame, wrap="word", state="disabled", height=20)
        self.output_text.pack(fill="both", expand=True)

        # Aggiungi una sezione di testo esplicativo nella parte bassa
        explanation_label = tk.Label(
            right_frame,
            text="EXPLANATION:\n"
            "1) E: Young's Modulus of the material.\n"
            "2) nu: Poisson's coefficent of the material.\n"
            "3) NUM_CORES: number of cores that you want to use to compute the solution.\n"
            "4) SOLVE: Start the solving procedure. The output on the right will keep you updated on the state of the simulation.\n" 
            "5) NEXT: proceed to the next step.\n" 
            "6) BACK: go back to the previous step.",
            font=("Arial", 12),
            wraplength=400,
            justify="left",
            fg="black",
            bg="#FFFACD"  # Giallo chiaro
        )
        explanation_label.pack(pady=20, anchor="s")  # Posiziona il testo nella parte bassa

        # Elenco degli step
        self.steps_titles = [
            "Step 1: Select .STEP file",
            "Step 2: Mesh Generation",
            "Step 3: Position Constraints",
            "Step 4: Pressure Constraints",
            "Step 5: Solver",
            "Step 6: Results Visualization"
        ]
        self.step_labels = []
        for i, title in enumerate(self.steps_titles):
            label = tk.Label(
                steps_frame,
                text=title,
                font=("Arial", 12),
                bg="lightgray",
                fg="black",
                anchor="w",
                padx=10
            )
            label.pack(fill="x", pady=5)
            self.step_labels.append(label)

        # Evidenzia lo step corrente
        self.highlight_current_step(4)

    def highlight_current_step(self, current_step):
        """Evidenzia lo step corrente nell'elenco degli step."""
        for i, label in enumerate(self.step_labels):
            if i == current_step:
                label.config(bg="yellow", font=("Arial", 10, "bold"))
            else:
                label.config(bg="lightgray", font=("Arial", 10))
    
    def show_notice_step5(self):
        # Mostra il messaggio di avviso quando si apre Step5
        messagebox.showinfo(
            "Notice",
            "Once you press 'Solve', the software might appear unresponsive. "
            "This is normal and due to the program's structure. Calculations will proceed normally."
        )

    def run_solver(self):
        # Get inputs
        e_value = self.e_entry.get()
        nu_value = self.nu_entry.get()
        num_cores = self.num_cores_entry.get()

        if not e_value or not nu_value or not num_cores:
            messagebox.showwarning("Warning", "Please fill in all fields.")
            return

        try:
            e_value = float(e_value)
            nu_value = float(nu_value)
            num_cores = int(num_cores)
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Ensure E and nu are numbers, and num_cores is an integer.")
            return
        
        if num_cores > multiprocessing.cpu_count():
            messagebox.showerror("Error", f"Number of cores exceeds available cores ({multiprocessing.cpu_count()}).")
            return

        # Run the solver script and capture its output
        self.output_text.config(state="normal")
        self.output_text.delete(1.0, tk.END)  # Clear previous output
        #self.output_text.insert(tk.END, "Running solver...\n")
        self.output_text.config(state="disabled")

        # Callback per scrivere nel widget
        def print_to_output(*args, sep=" ", end="\n"):
            text = sep.join(str(arg) for arg in args) + end
            self.output_text.config(state="normal")
            self.output_text.insert(tk.END, text)
            self.output_text.see(tk.END)
            self.output_text.config(state="disabled")
            self.output_text.update_idletasks()  # FORZA l'aggiornamento immediato


        # Funzione da eseguire in un thread
        def run_script():
            try:
                module = importlib.import_module("script.solver")
                module.main_solver(
                    self.controller.output_folder,
                    e_value, 
                    nu_value, 
                    num_cores,
                    print_func=print_to_output
                )
            except Exception as e:
                print_to_output(f"Error occurred: {e}")

        run_script()  # Esegui la funzione sul thread principale

        self.controller.solver_counter = 1
    
    def next_step_solver(self):
        # Controlla se il solver è stato eseguito con successo
        if self.controller.solver_counter == 0:
            messagebox.showwarning("Warning", "Please run the solver first.")
            return

        # Proceed to the next step
        self.controller.show_step(5)

class Step6(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.results_folder = None  # Variabile per memorizzare il percorso della cartella risultati
        self.selected_option = tk.StringVar(value="Select Variable")  # Variabile per il menu a tendina (default)

        # Layout: Left for controls, Right for displaying the selected folder path
        left_frame = tk.Frame(self, bg="white")
        left_frame.pack(side="left", fill="both", expand=False, padx=20, pady=20)

        steps_frame = tk.Frame(left_frame, bg="lightgray", width=200)
        steps_frame.pack(side="left", fill="y", padx=0, pady=0)

        main_frame = tk.Frame(left_frame, bg=None)
        main_frame.pack(side="left", fill="both", expand=True, padx=(10,0), pady=0)

        right_frame = tk.Frame(self, bg="lightgray")
        right_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        # Left Frame: Controls
        tk.Label(main_frame, text="Step 6: Visualizzazione Risultati", font=("Arial", 14)).pack(pady=10)

        # Button to select the results folder
        tk.Button(main_frame, text="Results Folder", command=self.select_results_folder).pack(pady=10)

        # Dropdown menu for selecting an option
        tk.Label(main_frame, text="Select Option:", font=("Arial", 12)).pack(pady=5)
        options = ["Von Mises Stresses (Elements)", "Von Mises Stresses (Nodes)", "Disp_x", "Disp_y", "Disp_z"]
        options_menu = tk.OptionMenu(main_frame, self.selected_option, *options)
        options_menu.pack(pady=5)

        # Button to view results
        tk.Button(main_frame, text="View Results", command=self.run_postprocessing).pack(pady=10)

        # Navigation buttons (in basso a sinistra)
        tk.Button(main_frame, text="Back", command=controller.previous_step).pack(pady=10)

        # Right Frame: Display the selected folder path
        tk.Label(right_frame, text="Selected Results Folder:", font=("Arial", 12)).pack(anchor="w", pady=5)
        self.folder_path_label = tk.Label(right_frame, text="No folder selected", fg="gray", wraplength=400, anchor="w", justify="left")
        self.folder_path_label.pack(anchor="w", pady=5)

        # Aggiungi una sezione di testo esplicativo nella parte bassa
        explanation_label = tk.Label(
            main_frame,
            text="EXPLANATION:\n"
            "1) RESULT FOLDER: folder in which the RESULTS Folder is located.\n"
            "2) SELECT OPTION: Select what you want to visualize on your 3d model.\n"
            "The options are:\n"
            "    - Stresses computed with VON MISES model shown in elements\n"
            "    - Stresses computed with VON MISES model computed in every node\n"
            "    - Displacement in x\n"
            "    - Displacement in y\n"
            "    - Displacement in z\n",
            font=("Arial", 12),
            wraplength=400,
            justify="left",
            fg="black",
            bg="#FFFACD"  # Giallo chiaro
        )
        explanation_label.pack(pady=20, anchor="s")  # Posiziona il testo nella parte bassa

        # Elenco degli step
        self.steps_titles = [
            "Step 1: Select .STEP file",
            "Step 2: Mesh Generation",
            "Step 3: Position Constraints",
            "Step 4: Pressure Constraints",
            "Step 5: Solver",
            "Step 6: Results Visualization"
        ]
        self.step_labels = []
        for i, title in enumerate(self.steps_titles):
            label = tk.Label(
                steps_frame,
                text=title,
                font=("Arial", 12),
                bg="lightgray",
                fg="black",
                anchor="w",
                padx=10
            )
            label.pack(fill="x", pady=5)
            self.step_labels.append(label)

        # Evidenzia lo step corrente
        self.highlight_current_step(5)

    def highlight_current_step(self, current_step):
        """Evidenzia lo step corrente nell'elenco degli step."""
        for i, label in enumerate(self.step_labels):
            if i == current_step:
                label.config(bg="yellow", font=("Arial", 10, "bold"))
            else:
                label.config(bg="lightgray", font=("Arial", 10))


    def select_results_folder(self):
        """Open a dialog to select the results folder."""
        folder = filedialog.askdirectory()
        if folder:
            self.results_folder = folder
            self.folder_path_label.config(text=folder, fg="black")  # Update the label with the selected folder path

    def run_postprocessing(self):
        """Run the postprocessing script with the selected folder and option."""
        if not self.results_folder:
            messagebox.showwarning("Warning", "Please select a results folder first.")
            return

        selected_option = self.selected_option.get()

        try:
            # Import dinamico dello script
            module = importlib.import_module("script.postprocessing")
            module.postprocess(self.results_folder, selected_option)
        except Exception as e:
            messagebox.showerror("Errore", f"Errore durante la visualizzazione: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = FEMApp()
    app.mainloop()

