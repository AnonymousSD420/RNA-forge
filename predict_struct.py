import pickle
import os
import numpy as np
from build_iter import *
import warnings
import json
import subprocess
import sys
import argparse
import tempfile
import shutil
from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.Atom import Atom as PDBAtom
from Bio import SeqIO

def write_pdb(structure, output_file, sequence):
    pdb_structure = Structure.Structure("RNA")
    model = Model.Model(0)
    chain = Chain.Chain("A")

    for i, residue in enumerate(structure):
        res_id = (" ", i + 1, " ")  # Residue ID (no insertion code)
        residue_obj = Residue.Residue(res_id, sequence[i], "")
        for atom_name, atom in residue.items():
            # Create Atom object
            pdb_atom = PDBAtom(
                name=atom_name,
                coord=atom['coord'],
                bfactor=0.00,  # Placeholder B-factor
                occupancy=1.00,
                altloc=" ",
                fullname=f"{atom_name:4}",
                serial_number=None,
                element=atom_name[0]  # Assumes the first letter of atom_name is the element
            )
            residue_obj.add(pdb_atom)
        chain.add(residue_obj)

    model.add(chain)
    pdb_structure.add(model)

    # Write the PDB file
    io = PDBIO()
    io.set_structure(pdb_structure)
    io.save(output_file)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Predict RNA 3D structure from FASTA file')
    parser.add_argument('--input_fasta', required=True,
                        help='Path to input FASTA file containing RNA sequence')
    parser.add_argument('--distance_matrix_predictor', choices=['rinalmo', 'birnabert'], 
                        default='birnabert',
                        help='Model to use for distance matrix prediction (default: birnabert)')
    parser.add_argument('--torsion_angle_predictor', choices=['rtb', 'drt'], 
                        default='rtb',
                        help='Model to use for torsion angle prediction (default: rtb)')
    parser.add_argument('--output_pdb_path', required=True,
                        help='Path to save the output PDB file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    return parser.parse_args()


def read_fasta_sequence(fasta_path):
    """Read the first sequence from FASTA file"""
    try:
        for record in SeqIO.parse(fasta_path, "fasta"):
            return str(record.seq).upper(), record.id
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        sys.exit(1)


def predict_distance_matrix(fasta_path, model_type, temp_dir):
    """Run distance_matrix_pred.py to predict distance matrix"""
    distance_matrix_path = os.path.join(temp_dir, 'distance_matrix.pkl')
    
    cmd = [
        sys.executable, 'distance_matrix_pred.py',
        '--model', model_type,
        '--fasta_path', fasta_path,
        '--output_path', distance_matrix_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error predicting distance matrix: {e}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)
    
    # Load the predicted distance matrix
    with open(distance_matrix_path, 'rb') as f:
        distance_matrix = pickle.load(f)
    
    return distance_matrix


def predict_torsion_angles_rtb(sequence, temp_dir, seq_id):
    """Predict torsion angles using TorsionBERT (rtb)"""
    torsion_angles_path = os.path.join(temp_dir, f'{seq_id}_torsion.json')
    
    # Check if RNA-TorsionBERT directory exists
    rtb_dir = 'RNA-TorsionBERT'
    if not os.path.exists(rtb_dir):
        print(f"Error: RNA-TorsionBERT directory not found at {rtb_dir}")
        sys.exit(1)
    
    # Save current directory and change to RNA-TorsionBERT
    current_dir = os.getcwd()
    os.chdir(rtb_dir)
    
    try:
        cmd = [
            'python', '-m', 'src.rna_torsionBERT_cli',
            f'--in_seq={sequence}',
            f'--out_path={os.path.join(current_dir, torsion_angles_path)}'
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error predicting torsion angles: {e}")
        os.chdir(current_dir)  # Return to original directory before exiting
        sys.exit(1)
    finally:
        # Always return to original directory
        os.chdir(current_dir)

    # Read the torsion angles
    try:
        with open(torsion_angles_path, 'r') as f:
            torsion_angles = json.load(f)
    except Exception as e:
        print(f"Error reading torsion angles file: {e}")
        sys.exit(1)

    return torsion_angles


def main():
    """Main function to predict RNA structure from FASTA"""
    warnings.filterwarnings("ignore")
    
    # Parse arguments
    args = parse_arguments()
    
    # Create temporary directory for intermediate files
    temp_dir = tempfile.mkdtemp(prefix='rna_structure_pred_')
    
    try:
        # Read sequence from FASTA
        sequence, seq_id = read_fasta_sequence(args.input_fasta)
        print(f"Processing sequence: {seq_id} (length: {len(sequence)})")
        
        # Step 1: Predict distance matrix
        print(f"Predicting distance matrix using {args.distance_matrix_predictor}...")
        distance_matrix = predict_distance_matrix(
            args.input_fasta, 
            args.distance_matrix_predictor, 
            temp_dir
        )
        
        # Step 2: Compute coordinates from distance matrix
        print("Computing 3D coordinates from distance matrix...")
        coordinates = compute_coordinates_from_distance_matrix(distance_matrix)
        
        # Step 3: Predict torsion angles
        print(f"Predicting torsion angles using {args.torsion_angle_predictor}...")
        if args.torsion_angle_predictor == 'rtb':
            torsion_angles = predict_torsion_angles_rtb(sequence, temp_dir, seq_id)
        elif args.torsion_angle_predictor == 'drt':
            # Placeholder for future implementation
            raise NotImplementedError("DRT torsion angle predictor not yet implemented")
        
        # Step 4: Build the 3D structure
        print(f"Building 3D RNA structure (seed: {args.seed})...")
        np.random.seed(args.seed)
        rna_structure = build_rna(sequence, coordinates, torsion_angles)
        
        # Step 5: Write the PDB file
        print(f"Writing PDB file to {args.output_pdb_path}...")
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output_pdb_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        write_pdb(rna_structure, args.output_pdb_path, sequence)
        
        print(f"Successfully generated PDB file: {args.output_pdb_path}")
        
    except Exception as e:
        print(f"Error during structure prediction: {e}")
        sys.exit(1)
    
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary files")


if __name__ == '__main__':
    main()