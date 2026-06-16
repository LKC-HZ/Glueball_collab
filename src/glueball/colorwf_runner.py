#!/usr/bin/env python3
"""
ColorFactor Runner - Run the Fortran color_wf program from Python and parse results.

Usage:
    from colorwf_runner import run_color_wf, parse_color_wf_output
    
    result = run_color_wf(n=1, m=1, k=1)
    print(result)
"""

import subprocess
import re
import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default path to the Fortran executable.
# Accepts both Windows paths (D:\...) and WSL paths (/mnt/d/... or /d/...).
# The script will auto-detect and convert as needed.
DEFAULT_EXECUTABLE = Path(
    "PATH")

# The input file the Fortran program needs in its CWD
REQUIRED_INPUT_FILE = "fabc.dat"






































# ---------------------------------------------------------------------------
# Data structures for parsed output
# ---------------------------------------------------------------------------
# this helps python decompose fortran output.
@dataclass
class WavefunctionComponent:
    """One non-zero component of a singlet wavefunction."""
    state_label: str          # e.g. "q1 | qb1 | g_C1"
    quark_colors: List[int]   # colors for each quark (1-based)
    antiquark_colors: List[int]  # colors for each antiquark (1-based)
    gluon_labels: List[str]   # gluon labels (CW or GM)
    coefficient: complex      # complex coefficient

@dataclass
class SingletState:
    """A single color-singlet state."""
    index: int                # 1-based singlet index
    basis: str                # "CW" or "GM"
    state_type: str           # e.g. "real", "imag", "d-type", "f-type", "GM"
    components: List[WavefunctionComponent] = field(default_factory=list)

@dataclass
class ColorWfResult:
    """Complete result from a color_wf run."""
    n: int
    m: int
    k: int
    theoretical_singlets: int = 0
    full_dimension: int = 0
    zero_weight_dimension: int = 0
    converged_eigenvalues: int = 0
    actual_singlets: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    cw_singlets: List[SingletState] = field(default_factory=list)
    gm_singlets: List[SingletState] = field(default_factory=list)
    raw_stdout: str = ""

























































# ---------------------------------------------------------------------------
# Path helpers (WSL detection / conversion)
# ---------------------------------------------------------------------------

def is_windows() -> bool:
    """Return True if running on Windows."""
    return sys.platform in ("win32", "cygwin")


def has_wsl() -> bool:
    """Check if WSL is available on this system."""
    return shutil.which("wsl") is not None


def _normalize_to_windows_path(p: Path) -> Path:
    """
    Convert a path that might be in WSL format (/mnt/d/... or /d/...)
    into a proper Windows Path.
    """
    p_str = str(p).replace("\\", "/")
    
    # Handle /mnt/d/... format -> D:/...
    if p_str.startswith("/mnt/"):
        drive = p_str[5]  # e.g. 'd' from '/mnt/d/...'
        rest = p_str[6:]  # after '/mnt/d'
        return Path(f"{drive.upper()}:{rest}")
    
    # Handle /d/... format (Git Bash) -> D:/...
    if len(p_str) >= 3 and p_str[0] == "/" and p_str[2] == "/":
        drive = p_str[1]
        if drive.isalpha():
            return Path(f"{drive.upper()}:{p_str[2:]}")
    
    return p


def windows_to_wsl_path(win_path: Path) -> str:
    r"""
    Convert a Windows path to a WSL /mnt/... path.
    
    Examples:
        D:\foo\bar -> /mnt/d/foo/bar
        C:\Users\X -> /mnt/c/Users/X
    """
    p = Path(win_path)
    # Get absolute path if possible
    try:
        p = p.resolve()
    except Exception:
        pass
    
    p_str = str(p)
    # Detect drive letter
    if len(p_str) >= 2 and p_str[1] == ":":
        drive = p_str[0].lower()
        rest = p_str[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    
    # Already POSIX-style (e.g. from Git Bash)
    p_str = p_str.replace("\\", "/")
    if p_str.startswith("/mnt/"):
        return p_str
    if len(p_str) >= 3 and p_str[0] == "/" and p_str[2] == "/":
        drive = p_str[1].lower()
        rest = p_str[2:]
        return f"/mnt/{drive}{rest}"
    
    # Fallback
    return p_str.replace("\\", "/")


def resolve_executable_path(
    exe_path: Optional[Path] = None
) -> Tuple[str, str, bool]:
    """
    Determine how to run the executable.
    
    Returns:
        (exe_path_str, working_dir, use_wsl)
        - exe_path_str: the executable path to pass to subprocess
        - working_dir: the working directory to use
        - use_wsl: whether to wrap the command with WSL
    """
    if exe_path is None:
        exe_path = DEFAULT_EXECUTABLE
    
    # Normalize the input path to Windows format if necessary
    exe_path = _normalize_to_windows_path(Path(exe_path))
    
    try:
        exe_path = exe_path.resolve()
    except Exception:
        # If resolve fails (e.g. path doesn't exist), use as-is
        pass
    
    working_dir = exe_path.parent
    
    # Check if the executable is a Linux ELF but we're on Windows
    needs_wsl = False
    if is_windows():
        if has_wsl():
            # Heuristic: if the file is a Linux ELF, use WSL
            try:
                with open(exe_path, "rb") as f:
                    magic = f.read(4)
                    if magic == b"\x7fELF":
                        needs_wsl = True
            except Exception:
                pass
            # If user explicitly wants WSL, or we detected ELF, use WSL
            if needs_wsl:
                exe_str = windows_to_wsl_path(exe_path)
                work_str = windows_to_wsl_path(working_dir)
                return exe_str, work_str, True
        else:
            raise RuntimeError(
                "The color_wf executable is a Linux binary but WSL is not available. "
                "Please run this script in WSL or rebuild the executable for Windows."
            )
    
    return str(exe_path), str(working_dir), False


















































# ---------------------------------------------------------------------------
# Running the executable
# ---------------------------------------------------------------------------

def run_color_wf(
    n: int,
    m: int,
    k: int,
    executable: Optional[Path] = None,
    timeout: float = 120.0,
) -> ColorWfResult:
    """
    Run the color_wf Fortran program with given (n, m, k) and return parsed results.
    
    Parameters:
        n: number of quarks
        m: number of antiquarks
        k: number of gluons
        executable: path to the color_wf executable (default: auto-detect)
        timeout: maximum seconds to wait for the program to finish
    
    Returns:
        ColorWfResult dataclass with all parsed data.
    """
    result = ColorWfResult(n=n, m=m, k=k)
    
    # Determine execution method
    exe_path, work_dir, use_wsl = resolve_executable_path(executable)
    
    # Verify fabc.dat exists in the working directory
    if use_wsl:
        # We can't directly check WSL filesystem, but we can check the Windows side
        if executable is None:
            executable = DEFAULT_EXECUTABLE
        win_work_dir = _normalize_to_windows_path(Path(executable)).parent
        win_fabc = win_work_dir / REQUIRED_INPUT_FILE
        if not win_fabc.exists():
            raise FileNotFoundError(
                f"Required input file '{REQUIRED_INPUT_FILE}' not found in {win_work_dir}. "
                "The Fortran program needs this file in its working directory."
            )
    else:
        fabc_path = Path(work_dir) / REQUIRED_INPUT_FILE
        if not fabc_path.exists():
            raise FileNotFoundError(
                f"Required input file '{REQUIRED_INPUT_FILE}' not found in {work_dir}. "
                "The Fortran program needs this file in its working directory."
            )
    
    # Build command
    if use_wsl:
        # Use WSL to run the Linux binary
        cmd = [
            "wsl", "-e", "bash", "-c",
            f'cd "{work_dir}" && "{exe_path}" {n} {m} {k}'
        ]
    else:
        cmd = [exe_path, str(n), str(m), str(k)]
    
    # Run the process
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=None if use_wsl else work_dir,
        )
    except subprocess.TimeoutExpired as exc:
        result.errors.append(f"Fortran program timed out after {timeout} seconds")
        result.raw_stdout = exc.stdout or ""
        return result
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Could not execute the Fortran program. Command: {cmd}\n"
            f"Original error: {exc}"
        ) from exc
    
    # Store raw output
    result.raw_stdout = proc.stdout
    
    # Check for process errors
    if proc.returncode != 0:
        result.errors.append(
            f"Fortran program exited with code {proc.returncode}.\n"
            f"STDERR: {proc.stderr.strip()}"
        )
        # Still try to parse whatever stdout we got
    
    # Parse the output
    parse_color_wf_output(proc.stdout, result)
    
    # Also capture any stderr warnings
    if proc.stderr:
        stderr_lines = proc.stderr.strip().splitlines()
        for line in stderr_lines:
            line = line.strip()
            if line and not line.startswith("Note:"):
                result.warnings.append(f"Fortran stderr: {line}")
    
    return result









































# ---------------------------------------------------------------------------
# Parsing the Fortran output
# ---------------------------------------------------------------------------

# Regex patterns for the output lines
RE_HEADER_SINGLET = re.compile(r"---\s+Singlet\s+(\d+)\s+\(([^)]+)\)\s*---")
RE_GM_HEADER = re.compile(r"=====\s+Gell-Mann\s+basis\s+representation\s+=====")
RE_STATE_LINE = re.compile(
    r"^(.*?)\s*:\s+([+-]?\d+\.\d+)\s+([+-]\d+\.\d+)i$"
)
RE_DIAGNOSTIC = re.compile(
    r"^\s*(Theoretical singlets|Full dimension|Zero-weight dimension|"
    r"Converged eigenvalues|Actual number of singlets)\s*:\s*(\d+)"
)
RE_WARNING = re.compile(r"WARNING[:\s](.*)", re.IGNORECASE)


def parse_color_wf_output(stdout: str, result: ColorWfResult) -> None:
    """
    Parse the stdout from color_wf into the result object.
    """
    lines = stdout.splitlines()
    
    current_section = "CW"      # "CW" or "GM"
    current_singlet: Optional[SingletState] = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for diagnostics
        diag_match = RE_DIAGNOSTIC.match(line)
        if diag_match:
            key = diag_match.group(1)
            val = int(diag_match.group(2))
            if key == "Theoretical singlets":
                result.theoretical_singlets = val
            elif key == "Full dimension":
                result.full_dimension = val
            elif key == "Zero-weight dimension":
                result.zero_weight_dimension = val
            elif key == "Converged eigenvalues":
                result.converged_eigenvalues = val
            elif key == "Actual number of singlets":
                result.actual_singlets = val
            continue
        
        # Check for warnings in stdout
        warn_match = RE_WARNING.search(line)
        if warn_match:
            result.warnings.append(warn_match.group(1).strip())
            continue
        
        # Check for GM basis header
        if RE_GM_HEADER.match(line):
            current_section = "GM"
            continue
        
        # Check for singlet header
        singlet_match = RE_HEADER_SINGLET.match(line)
        if singlet_match:
            idx = int(singlet_match.group(1))
            state_type = singlet_match.group(2).strip()
            
            new_singlet = SingletState(
                index=idx,
                basis=current_section,
                state_type=state_type,
                components=[],
            )
            
            if current_section == "CW":
                result.cw_singlets.append(new_singlet)
            else:
                result.gm_singlets.append(new_singlet)
            
            current_singlet = new_singlet
            continue
        
        # Check for state component line
        state_match = RE_STATE_LINE.match(line)
        if state_match and current_singlet is not None:
            label = state_match.group(1).strip()
            real_part = float(state_match.group(2))
            imag_part = float(state_match.group(3))
            coeff = complex(real_part, imag_part)
            
            # Parse the state label into parts
            quark_colors, antiquark_colors, gluon_labels = parse_state_label(
                label, result.n, result.m, result.k
            )
            
            component = WavefunctionComponent(
                state_label=label,
                quark_colors=quark_colors,
                antiquark_colors=antiquark_colors,
                gluon_labels=gluon_labels,
                coefficient=coeff,
            )
            current_singlet.components.append(component)
            continue
        
        # Unrecognized line -- could be noise, ARPACK messages, etc.
        # We silently ignore most of them, but keep known error messages
        if "error" in line.lower() or "stop" in line.lower():
            result.errors.append(f"Fortran output: {line}")
    
    # Consistency checks
    if result.actual_singlets != result.theoretical_singlets:
        result.warnings.append(
            f"Mismatch: actual_singlets ({result.actual_singlets}) != "
            f"theoretical_singlets ({result.theoretical_singlets})"
        )


def parse_state_label(
    label: str, n: int, m: int, k: int
) -> Tuple[List[int], List[int], List[str]]:
    """
    Parse a state label like "q1,q2 |qb1,qb2 |g_C1,g(+1,0)" into
    quark colors, antiquark colors, and gluon labels.
    
    Note: Fortran uses ' |' (space-pipe) as separator, not ' | ' (space-pipe-space),
    because it calls trim() before appending each section, stripping the trailing space.
    """
    quark_colors: List[int] = []
    antiquark_colors: List[int] = []
    gluon_labels: List[str] = []
    
    # Fortran output uses ' |' as separator (space before pipe, no space after)
    # because trim() removes the trailing space before the next append.
    parts = [p.strip() for p in label.split(" |")]
    parts = [p for p in parts if p]  # remove empty strings
    
    # Determine which part is which based on n, m, k
    part_idx = 0
    
    # Extract quarks
    if n > 0 and part_idx < len(parts):
        q_part = parts[part_idx]
        for token in q_part.split(","):
            token = token.strip()
            if token.startswith("q") and not token.startswith("qb"):
                try:
                    quark_colors.append(int(token[1:]))
                except ValueError:
                    pass
        part_idx += 1
    
    # Extract antiquarks
    if m > 0 and part_idx < len(parts):
        qb_part = parts[part_idx]
        for token in qb_part.split(","):
            token = token.strip()
            if token.startswith("qb"):
                try:
                    antiquark_colors.append(int(token[2:]))
                except ValueError:
                    pass
        part_idx += 1
    
    # Extract gluons
    if k > 0 and part_idx < len(parts):
        g_part = parts[part_idx]
        for token in g_part.split(","):
            token = token.strip()
            if token and not token.startswith("q") and not token.startswith("qb"):
                gluon_labels.append(token)
    
    return quark_colors, antiquark_colors, gluon_labels












































# ---------------------------------------------------------------------------
# Convenience accessors for numerical results
# ---------------------------------------------------------------------------

def get_singlet_coefficients(
    result: ColorWfResult,
    basis: str = "CW",
    singlet_index: int = 1,
) -> Dict[str, complex]:
    """
    Return a flat dict {state_label: coefficient} for a specific singlet.
    
    Parameters:
        result: parsed ColorWfResult
        basis: "CW" or "GM"
        singlet_index: 1-based index of the singlet
    """
    singlets = result.cw_singlets if basis.upper() == "CW" else result.gm_singlets
    for singlet in singlets:
        if singlet.index == singlet_index:
            return {comp.state_label: comp.coefficient for comp in singlet.components}
    return {}


def get_singlet_array(
    result: ColorWfResult,
    basis: str = "CW",
    singlet_index: int = 1,
) -> Tuple[np.ndarray, List[str]]:
    """
    Return a numpy array of coefficients and corresponding state labels.
    
    Returns:
        coeffs: 1D complex numpy array
        labels: list of state label strings
    """
    coeffs = get_singlet_coefficients(result, basis, singlet_index)
    if not coeffs:
        return np.array([], dtype=complex), []
    labels = list(coeffs.keys())
    values = np.array([coeffs[l] for l in labels], dtype=complex)
    return values, labels












































# ---------------------------------------------------------------------------
# Main script (for direct execution)
# ---------------------------------------------------------------------------

def main():
    """Example usage when running this script directly."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run the color_wf Fortran program and parse results."
    )
    parser.add_argument("n", type=int, help="Number of quarks")
    parser.add_argument("m", type=int, help="Number of antiquarks")
    parser.add_argument("k", type=int, help="Number of gluons")
    parser.add_argument(
        "--exe", "-e", type=Path, default=None,
        help="Path to the color_wf executable (default: auto-detect)"
    )
    parser.add_argument(
        "--timeout", "-t", type=float, default=120.0,
        help="Timeout in seconds (default: 120)"
    )
    parser.add_argument(
        "--save-raw", "-r", type=Path, default=None,
        help="Save raw stdout to a file"
    )
    
    args = parser.parse_args()
    
    print(f"Running color_wf with n={args.n}, m={args.m}, k={args.k} ...")
    print(f"Executable: {args.exe or DEFAULT_EXECUTABLE}")
    print("-" * 60)
    
    result = run_color_wf(args.n, args.m, args.k, executable=args.exe, timeout=args.timeout)
    
    # Save raw output if requested
    if args.save_raw:
        with open(args.save_raw, "w", encoding="utf-8") as f:
            f.write(result.raw_stdout)
        print(f"Raw stdout saved to: {args.save_raw}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Theoretical singlets:       {result.theoretical_singlets}")
    print(f"Full dimension:             {result.full_dimension}")
    print(f"Zero-weight dimension:      {result.zero_weight_dimension}")
    print(f"Converged eigenvalues:      {result.converged_eigenvalues}")
    print(f"Actual singlets found:      {result.actual_singlets}")
    print(f"CW singlets parsed:         {len(result.cw_singlets)}")
    print(f"GM singlets parsed:         {len(result.gm_singlets)}")
    
    if result.warnings:
        print(f"\n{'='*60}")
        print("WARNINGS")
        print(f"{'='*60}")
        for w in result.warnings:
            print(f"  - {w}")
    
    if result.errors:
        print(f"\n{'='*60}")
        print("ERRORS")
        print(f"{'='*60}")
        for e in result.errors:
            print(f"  - {e}")
    
    # Print detailed wavefunctions
    for basis_name, singlets in [("Cartan-Weyl", result.cw_singlets),
                                   ("Gell-Mann", result.gm_singlets)]:
        if not singlets:
            continue
        print(f"\n{'='*60}")
        print(f"{basis_name} BASIS")
        print(f"{'='*60}")
        for singlet in singlets:
            print(f"\n--- Singlet {singlet.index} ({singlet.state_type}) ---")
            for comp in singlet.components:
                c = comp.coefficient
                print(f"  {comp.state_label:40s} : {c.real:+.8f} {c.imag:+.8f}i")
    
    # Show numpy array example
    print(f"\n{'='*60}")
    print("NUMPY ARRAY ACCESS (example)")
    print(f"{'='*60}")
    if result.cw_singlets:
        coeffs, labels = get_singlet_array(result, basis="CW", singlet_index=1)
        print(f"CW Singlet 1 coefficients shape: {coeffs.shape}")
        print(f"dtype: {coeffs.dtype}")
        print(f"First 5 entries:")
        for i in range(min(5, len(coeffs))):
            print(f"  {labels[i]:40s} -> {coeffs[i]}")
    
    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}")
    
    return result


if __name__ == "__main__":
    main()