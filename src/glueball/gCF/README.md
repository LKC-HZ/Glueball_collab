# Gluon-to-Gluon Color Factor Calculator

## Overview

This program calculates the **color factors** for multi-gluon processes of the form

$$N \text{ gluons} \rightarrow M \text{ gluons}$$

in QCD. It enumerates all possible diagram configurations via the `allvertices` module, computes the SU(3) color-space contraction for each configuration using `np.einsum`, and reads color-singlet wavefunctions from a Fortran program (`color_wf.f90`) via a Python wrapper.

---

## Repository Structure

| File | Role |
|------|------|
| `main.py` | Entry point. Prompts for number of incoming/outgoing gluons, generates vertex plans, loops over all singlet types, and prints/saves the final color factors. |
| `allvertices.py` | Generates all valid vertex configurations for a given $N \rightarrow M$ process. Outputs a list of plans, each containing `incoming`/`outgoing` gluon IDs and vertex types (`3-gluon`, `4-gluon`, or `spectator`). |
| `gluon_to_gluon_color_factor.py` | Core computation module. Defines the 3-gluon and 4-gluon vertex tensors, reads color-singlet wavefunctions, and performs the `einsum` color-space contraction for a single vertex plan. Uses `lru_cache` to avoid repeated Fortran invocations. |
| `colorwf_runner.py` | Python wrapper for the Fortran program `color_wf`. Runs the executable via subprocess (with WSL support on Windows), parses stdout into `ColorWfResult` dataclasses, and provides `get_singlet_array()` for numerical access. |
| `su3constants.py` | SU(3) group-theory constants. Contains Gell-Mann matrices, fundamental generators $T^a$, adjoint generators $F^a$, antisymmetric structure constants $f^{abc}$, symmetric $d^{abc}$, and Casimir invariants $C_F$, $C_A$, $T_F$. |
| `color_wf` (Fortran) | External executable (not in this folder). Computes color-singlet wavefunctions in the Cartan-Weyl and Gell-Mann bases. Expects a `fabc.dat` input file in its working directory. |

---

## Program Flow

```
main.py
   ├─ prompts for N (incoming) and M (outgoing) gluons
   ├─ calls colorwf_runner.run_color_wf(0,0,N)  →  reads all incoming singlets
   ├─ calls colorwf_runner.run_color_wf(0,0,M)  →  reads all outgoing singlets
   ├─ calls allvertices.generate_vertex_plans(N,M)
   └─ triple loop:
         for each incoming singlet type (d/f)
           for each outgoing singlet type (d/f)
             for each vertex plan
                → calls gluon_contraction_by_plan(...)
                    ├─ reads wavefunctions (cached)
                    ├─ builds 3- or 4-gluon vertex tensors from su3.fabc
                    ├─ assembles einsum operands with index convention:
                       incoming gluon i  →  -i  (conceptual)
                       outgoing gluon j  →  +j  (conceptual)
                    └─ returns scalar color factor + singlet metadata
                → prints to console (commented-out)
                → writes one line to txt
```



---

## Running Conditions & Prerequisites

### 1. Python Environment
- Python 3.9+ recommended
- NumPy (required for `einsum` and array operations)
- Standard library only otherwise (`functools`, `pathlib`, `subprocess`, `re`, `itertools`)

### 2. Fortran Executable (`color_wf`)
The wavefunction data comes from a pre-compiled Fortran program (`color_wf.f90`).

- **On Windows:** The executable is a Linux ELF binary. The wrapper auto-detects this and routes the call through **WSL** (Windows Subsystem for Linux). You must have WSL installed and `wsl` available in your PATH.


### 3. Required Input File
The Fortran program expects a file named **`fabc.dat`** in its **working directory** (the same folder as the `color_wf` executable). If this file is missing, `colorwf_runner` will raise a `FileNotFoundError`.

### 4. Path Configuration 
Inside `colorwf_runner.py`, line 31–33:
```python
DEFAULT_EXECUTABLE = Path(
    "PATH")
```
Update this path to point to your actual `color_wf`.

---

## Usage

### Interactive Run
```bash
python main.py
```
Then enter the number of incoming and outgoing gluons when prompted. Example:
```
# of incoming gluons: 2
# of outgoing gluons: 3
```

### Output
- **Console:** Human-readable results showing the color factor, vertex plan ID, and singlet types (d-type / f-type) for each combination. (COMMENTED-OUT)
- **Text file:** e.g., `3_gluon_to_4_gluon_Color_Factor`, with one line per result.

---
