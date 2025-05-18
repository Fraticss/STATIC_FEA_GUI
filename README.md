# STATIC_FEA_GUI

**STATIC_FEA_GUI** is a graphical tool designed to perform **static structural analysis** on 3D models using the **Finite Element Method (FEM)**. It works with `.STEP` files and allows users to carry out the full analysis workflow — from mesh generation to post-processing — through a simplified interface. The computations are fully handled in **Python** using only open-source libraries, and the solution process leverages **parallel computing** to reduce processing times.

This application is particularly suitable for engineering students, researchers, or developers interested in learning or performing basic FEM simulations without relying on commercial tools.

---

## 🖥️ About the Interface

This is the **first public release** of the software, and the interface is intentionally kept **minimalistic and essential** to focus on core functionality. Despite its simplicity, it provides all the necessary tools to execute a complete static FEM analysis on isotropic materials.

The software is structured into **multiple steps**, each corresponding to a specific phase of the analysis:

1. **Import geometry**
2. **Generate mesh**
3. **Apply boundary conditions**
4. **Run the solver**
5. **Post-process results**

Each step includes a **yellow instruction box** explaining the available commands and how to interact with the current panel. This makes it easy to follow the workflow, even for users new to the tool.

---

## 📁 Repository Contents

This repository includes:
- The **Python source scripts** used to build the software.
- The **PyInstaller build files** used to generate the standalone executable.
- The actual **Windows executable** (`.exe`), available in the [Releases](https://github.com/Fraticss/STATIC_FEA_GUI/releases) section.

---

## 🚀 How to Use

1. Download the `.exe` from the [Releases](https://github.com/Fraticss/STATIC_FEA_GUI/releases) section.
2. Place your `.STEP` file in the `Input` folder created by the software.
3. Follow the on-screen steps, reading the instructions at each phase.
4. View your results in the final post-processing step.

---

## 💬 Feedback & Contributions

If you encounter any bugs or have ideas for improvements, feel free to open an issue or contact the developer directly. Feedback is highly appreciated and will help shape future versions of the software.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

Enjoy your finite element analysis!

