import numpy as np
import json
from scipy.linalg import eigh
from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.Atom import Atom as PDBAtom
from sklearn.manifold import MDS
from scipy.optimize import minimize

amp_bond_lenghts_angles = {}
gmp_bond_lenghts_angles = {}
ump_bond_lenghts_angles = {}
cmp_bond_lenghts_angles = {}

# O3_prime_P_O5_prime_bond_angle = 103.09630067626846
O3_prime_P_O5_prime_bond_angle_dict = {"mean": 103.71694259802273, "std_dev": 3.2407841777974657}
O3_prime_next_P_prime_distance = 0

data_directory = "./data_helper"


"""
All the utility vector operations and rigid body transformation funtions
"""
def get_rotation_matrix_angle(axis, theta):
    """
    Returns the rotation matrix that rotates a vector by `theta` degrees 
    anti-clockwise around the given `axis`.
    """
    # Convert degrees to radians
    theta = np.radians(theta)
    
    # Normalize the axis vector
    axis = axis / np.linalg.norm(axis)
    
    # Compute components of the rotation matrix using Rodrigues' formula
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    one_minus_cos = 1 - cos_theta
    x, y, z = axis
    R = np.array([
        [cos_theta + x*x*one_minus_cos, x*y*one_minus_cos - z*sin_theta, x*z*one_minus_cos + y*sin_theta],
        [y*x*one_minus_cos + z*sin_theta, cos_theta + y*y*one_minus_cos, y*z*one_minus_cos - x*sin_theta],
        [z*x*one_minus_cos - y*sin_theta, z*y*one_minus_cos + x*sin_theta, cos_theta + z*z*one_minus_cos]
    ])
    return R

def get_rotation_matrix_vector(vector1, vector2):
    """
    Returns the rotation matrix that rotates `vector1` to align with `vector2`.
    """
    # Normalize the vectors
    v1 = vector1 / np.linalg.norm(vector1)
    v2 = vector2 / np.linalg.norm(vector2)
    
    # Compute the rotation axis and angle
    axis = np.cross(v1, v2)
    if np.linalg.norm(axis) == 0:
        # Vectors are parallel, return identity matrix
        if np.dot(v1, v2) > 0:
            return np.eye(3)  # Same direction
        else:
            # Opposite direction: rotate 180 degrees around any perpendicular axis
            axis = np.array([1, 0, 0]) if abs(v1[0]) < 1 else np.array([0, 1, 0])
    else:
        axis = axis / np.linalg.norm(axis)
    
    # Calculate the angle
    theta = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
    
    # Use the first function to get the rotation matrix
    return get_rotation_matrix_angle(axis, theta)

def get_direction(vector1, vector2):
    """
    Returns the unit vector in the direction from `vector1` to `vector2`.
    """
    direction = vector2 - vector1
    return direction / np.linalg.norm(direction)

def get_distance(vector1, vector2):
    """
    Returns the Euclidean distance between `vector1` and `vector2`.
    """
    return np.linalg.norm(vector2 - vector1)

def get_angle(vector1, vector2):
    """
    Returns the angle in degrees between `vector1` and `vector2`.
    """
    dot_product = np.dot(vector1, vector2)
    cos_theta = dot_product / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def compute_point_D(A, B, C, BD, angle_ABD, angle_CBD, side="left"):
    """
    Compute the coordinates of point D in 3D space based on given inputs.

    Parameters:
    A, B, C: numpy arrays representing points in 3D space (e.g., np.array([x, y, z])).
    BD: Distance between points B and D.
    angle_ABD: Angle between vectors B->A and B->D (in degrees).
    angle_CBD: Angle between vectors B->C and B->D (in degrees).
    side: 'left' or 'right', determines which side of the plane D lies on.

    Returns:
    D: The coordinates of point D as a numpy array.
    """
    # Convert angles to radians
    angle_ABD = np.radians(angle_ABD)
    angle_CBD = np.radians(angle_CBD)
    
    # Compute vectors BA and BC
    BA = A - B
    BC = C - B
    
    # Normalize BA and BC
    BA = BA / np.linalg.norm(BA)
    BC = BC / np.linalg.norm(BC)
    
    # Compute the normal vector to the plane containing BA and BC
    n = np.cross(BA, BC)
    n = n / np.linalg.norm(n)  # Normalize
    
    # Compute a vector orthogonal to BA and n in the plane of BA and BC
    v = np.cross(n, BA)
    v = v / np.linalg.norm(v)  # Normalize
    
    # Spherical coordinates to Cartesian for D
    cos_ABD = np.cos(angle_ABD)
    sin_ABD = np.sin(angle_ABD)
    cos_CBD = np.cos(angle_CBD)
    sin_CBD = np.sin(angle_CBD)
    
    # Compute displacement vector d for D
    if side == "left":
        d = BD * (cos_ABD * BA + sin_ABD * cos_CBD * v + sin_ABD * sin_CBD * n)
    elif side == "right":
        d = BD * (cos_ABD * BA + sin_ABD * cos_CBD * v - sin_ABD * sin_CBD * n)
    else:
        raise ValueError("Invalid side: choose 'left' or 'right'")
    
    # Compute D as B + d
    D = B + d
    return D

def rigid_body_transformation(mapping_points, vertices):
    """
    Computes a rigid body transformation (rotation + translation) to map three source points 
    to three target points and applies the transformation to all vertices.

    Parameters:
        mapping_points (list of tuples): A list of 3 tuples, where each tuple is
            (target_coordinate, source_index).
            Example: [(np.array([x1, y1, z1]), 'A'), (np.array([x2, y2, z2]), 'B'), (np.array([x3, y3, z3]), 'C')].
        vertices (dict): A dictionary containing the original coordinates of the rigid body's vertices.
            Keys are vertex names, and values are numpy arrays of shape (3,).

    Returns:
        dict: A dictionary containing the transformed coordinates of all vertices, keyed by the same vertex names.
              The transformed coordinates are numpy arrays of shape (3,).
    """
    # Extract target points and source points
    target_coords = np.vstack([mp[0] for mp in mapping_points])  # Target points, shape (3, 3)
    source_coords = np.vstack([vertices[mp[1]] for mp in mapping_points])  # Source points, shape (3, 3)

    # Compute the centroids of both point sets
    source_centroid = np.mean(source_coords, axis=0)
    target_centroid = np.mean(target_coords, axis=0)

    # Center the points around the centroids
    source_centered = source_coords - source_centroid
    target_centered = target_coords - target_centroid

    # Compute the rotation matrix using SVD
    H = np.dot(source_centered.T, target_centered)  # Covariance matrix
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)  # Rotation matrix

    # Ensure a proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute the translation vector
    t = target_centroid - np.dot(R, source_centroid)

    # Apply the transformation to all vertices
    transformed_vertices = {}
    for key, coord in vertices.items():
        transformed_coord = np.dot(R, coord) + t  # Apply rotation and translation
        transformed_vertices[key] = transformed_coord

    return transformed_vertices

def rotate_rigid_body(rigid_body, pivot_key, axis, theta):
    """
    Rotates a rigid body around a pivot point counterclockwise by an angle theta.

    Parameters:
        rigid_body (dict): Dictionary of points {key: {'coord': [x, y, z], ...}}.
        pivot_key (str): The key of the pivot point.
        axis (list or tuple): The axis of rotation [x, y, z].
        theta (float): The rotation angle in degrees.

    Returns:
        dict: Dictionary of rotated rigid body points {key: {'coord': [x, y, z], ...}}.
    """
    # Convert theta to radians
    theta_rad = np.radians(theta)

    # Normalize the axis of rotation
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)

    # Rotation matrix using Rodrigues' rotation formula
    ux, uy, uz = axis
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    R = np.array([
        [cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta],
        [uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta],
        [uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]
    ])

    # Pivot point
    pivot = np.array(rigid_body[pivot_key]['coord'])

    # Rotate each point
    rotated_body = {}
    for key, data in rigid_body.items():
        coords = np.array(data['coord'])
        rotated_coords = np.dot(R, coords - pivot) + pivot
        # Preserve other attributes of the rigid body
        rotated_body[key] = {**data, 'coord': np.array(rotated_coords.tolist())}

    return rotated_body



def parse_pdb_to_dict(pdb_file):
    """
    Reads a PDB file and extracts the coordinates of each atom, 
    storing them in a dictionary keyed by the atom name.
    
    Args:
        pdb_file (str): Path to the PDB file.

    Returns:
        dict: A dictionary with atom names as keys and coordinates as values.
    """
    atom_coords = {}
    
    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                parts = line.split()
                atom_name = parts[2].strip()  # Atom name
                x, y, z = map(float, parts[6:9])  # Coordinates
                atom_coords[atom_name] = np.array([x, y, z])
    
    return atom_coords


def initialize():
    # load bond lengths and angles from json file
    with open(data_directory + '/A_properties_stats.json', 'r') as file:
        amp_properties = json.load(file)
    for key in amp_properties:
        amp_bond_lenghts_angles[key] = amp_properties[key]

    with open(data_directory + '/G_properties_stats.json', 'r') as file:
        gmp_properties = json.load(file)
    for key in gmp_properties:
        gmp_bond_lenghts_angles[key] = gmp_properties[key]

    with open(data_directory + '/U_properties_stats.json', 'r') as file:
        ump_properties = json.load(file)
    for key in ump_properties:
        ump_bond_lenghts_angles[key] = ump_properties[key]

    with open(data_directory + '/C_properties_stats.json', 'r') as file:
        cmp_properties = json.load(file)
    for key in cmp_properties:
        cmp_bond_lenghts_angles[key] = cmp_properties[key]


"""
This function is used to generate the Phosphorus (P) backbone coordinates from distance matrix.
"""
def compute_coordinates_from_distance_matrix(distance_matrix):
    """
    Compute the 3D coordinates of points given a distance matrix using multidimensional scaling.

    Args:
        distance_matrix (numpy.ndarray): n x n distance matrix.

    Returns:
        numpy.ndarray: n x 3 array of computed coordinates.
    """
    n = distance_matrix.shape[0]
    
    # Ensure the matrix is squared and symmetric
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Distance matrix must be square.")
    
    if not np.allclose(distance_matrix, distance_matrix.T, atol=1e-10):
        raise ValueError("Distance matrix must be symmetric.")
    
    # Centering matrix
    H = np.eye(n) - (1 / n) * np.ones((n, n))
    
    # Squared distance matrix
    D_squared = distance_matrix ** 2
    
    # Double centering the distance matrix
    B = -0.5 * H @ D_squared @ H
    
    # Eigenvalue decomposition
    eigvals, eigvecs = eigh(B)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Take the top 3 eigenvalues and corresponding eigenvectors
    L = np.diag(np.sqrt(eigvals[:3]))
    V = eigvecs[:, :3]
    
    # Compute the coordinates
    coordinates = V @ L
    
    return coordinates



def sample_value(dict):
    mean = dict['mean']
    std = dict['std_dev']
    return np.random.normal(mean, std)


def generate_atom(prev_coord, bond_length_dict, bond_angle_dict, torsion_angle, prev_vector, prev_right_vector):

    bond_length = sample_value(bond_length_dict)
    bond_angle = sample_value(bond_angle_dict)

    rotation_matrix_bond_angle = get_rotation_matrix_angle(prev_right_vector, 180 - bond_angle)
    rotation_matrix_torsion_angle = get_rotation_matrix_angle(prev_vector, torsion_angle)

    bond_angle_rotated_direction = np.dot(rotation_matrix_bond_angle, prev_vector)
    torsion_angle_rotated_direction = np.dot(rotation_matrix_torsion_angle, bond_angle_rotated_direction)

    new_right_vector = np.dot(rotation_matrix_torsion_angle, prev_right_vector)

    new_coord = prev_coord + bond_length * torsion_angle_rotated_direction

    return new_coord, new_right_vector  
    

def build_residue_coords(p_coord, torsion_angles, prev_vector, prev_right_vector, sequence, residue_idx, prev_residue=None):
    residue = {}
    residue['P'] = {"coord":p_coord, "residue_idx":residue_idx}

    current_point = p_coord

    bond_lengths_angles = {}
    ribose_sugar = {}
    n_base = {}
    if sequence[residue_idx] == 'A':
        bond_lengths_angles = amp_bond_lenghts_angles
        ribose_sugar = parse_pdb_to_dict(data_directory + '/ribose_sugar_A.pdb')
        n_base = parse_pdb_to_dict(data_directory + '/Adenine.pdb')
    elif sequence[residue_idx] == 'G':
        bond_lengths_angles = gmp_bond_lenghts_angles
        ribose_sugar = parse_pdb_to_dict(data_directory + '/ribose_sugar_G.pdb')
        n_base = parse_pdb_to_dict(data_directory + '/Guanine.pdb')
    elif sequence[residue_idx] == 'U':
        bond_lengths_angles = ump_bond_lenghts_angles
        ribose_sugar = parse_pdb_to_dict(data_directory + '/ribose_sugar_U.pdb')
        n_base = parse_pdb_to_dict(data_directory + '/Uracil.pdb')
    elif sequence[residue_idx] == 'C':
        bond_lengths_angles = cmp_bond_lenghts_angles
        ribose_sugar = parse_pdb_to_dict(data_directory + '/ribose_sugar_C.pdb')
        n_base = parse_pdb_to_dict(data_directory + '/Cytosine.pdb')

    # move "P-O5'" length in the direction of prev_vector to get O5'

    o5_prime = None
    if residue_idx == 0:
        o5_prime = current_point + sample_value(bond_lengths_angles["bond_lengths"]["P-O5'"]) * prev_vector
        new_right_vector = prev_right_vector
    else:  # prev_residue is not None
        o5_prime, new_right_vector = generate_atom(
            current_point, bond_lengths_angles["bond_lengths"]["P-O5'"], O3_prime_P_O5_prime_bond_angle_dict , torsion_angles['zeta'][residue_idx-1], prev_vector, prev_right_vector)
    
    
    
    residue['O5\''] = {"coord":o5_prime, "residue_idx":residue_idx}
    current_point = o5_prime

    # generate C5' atom
    c5_prime, new_right_vector = generate_atom(
        current_point, bond_lengths_angles["bond_lengths"]["O5'-C5'"], bond_lengths_angles["bond_angles"]["P-O5'-C5'"], torsion_angles['alpha'][residue_idx], prev_vector, new_right_vector
    )
    residue['C5\''] = {"coord":c5_prime, "residue_idx":residue_idx}
    current_point = c5_prime

    # generate C4' atom
    c4_prime, new_right_vector = generate_atom(
        current_point, bond_lengths_angles["bond_lengths"]["C5'-C4'"], bond_lengths_angles["bond_angles"]["O5'-C5'-C4'"], torsion_angles['beta'][residue_idx], prev_vector, new_right_vector
    )
    residue['C4\''] = {"coord":c4_prime, "residue_idx":residue_idx}
    current_point = c4_prime

    # generate C3' atom
    c3_prime, new_right_vector = generate_atom(
        current_point, bond_lengths_angles["bond_lengths"]["C4'-C3'"], bond_lengths_angles["bond_angles"]["C5'-C4'-C3'"], torsion_angles['gamma'][residue_idx], prev_vector, new_right_vector
    )

    # get O4' atom
    key1 = "C3'-C4'-O4'"   # angle CBD
    key2 = "C5'-C4'-O4'"   # angle ABD
    
    if key1 not in bond_lengths_angles["bond_angles"]:
        key1 = "O4'-C4'-C3'"
    if key2 not in bond_lengths_angles["bond_angles"]:
        key2 = "O4'-C4'-C3'"

    o4_prime = compute_point_D(c5_prime, c4_prime, c3_prime, sample_value(bond_lengths_angles["bond_lengths"]["C4'-O4'"]), sample_value(bond_lengths_angles["bond_angles"][key2]), sample_value(bond_lengths_angles["bond_angles"][key1]), "left")
    residue['O4\''] = {"coord":o4_prime, "residue_idx":residue_idx}

    residue['C3\''] = {"coord":c3_prime, "residue_idx":residue_idx}
    current_point = c3_prime

    # generate O3' atom
    o3_prime, new_right_vector = generate_atom(
        current_point, bond_lengths_angles["bond_lengths"]["C3'-O3'"], bond_lengths_angles["bond_angles"]["C4'-C3'-O3'"], torsion_angles['delta'][residue_idx], prev_vector, new_right_vector
    )
    residue['O3\''] = {"coord":o3_prime, "residue_idx":residue_idx}
    current_point = o3_prime

    # now we go for sugar part
    arranged_ribose_sugar_coords = rigid_body_transformation([(c3_prime,"C3'"),(c4_prime,"C4'"),(o4_prime,"O4'")], ribose_sugar)
    for key in arranged_ribose_sugar_coords:
        if key == "C3'" or key == "C4'" or key == "O4'":
            continue
        residue[key] = {"coord":arranged_ribose_sugar_coords[key], "residue_idx":residue_idx}

    # # now we go for base part
    direction_vector_o4_prime_2_c1_prime = get_direction(o4_prime, residue['C1\'']['coord'])

    if sequence[residue_idx] == 'A' or sequence[residue_idx] == 'G':  # purine
        direction_vector_c1_prime_2_n9 = get_direction(residue['C1\'']['coord'], residue['N9']['coord'])
        right_vector = np.cross(direction_vector_o4_prime_2_c1_prime, direction_vector_c1_prime_2_n9)
        right_vector = right_vector / np.linalg.norm(right_vector)

        # get C4 atom
        c4, _ = generate_atom(
            residue['N9']['coord'], bond_lengths_angles["bond_lengths"]["C4-N9"], bond_lengths_angles["bond_angles"]["C1'-N9-C4"], torsion_angles['chi'][residue_idx], direction_vector_c1_prime_2_n9, right_vector)
        
        arranged_n_base_coords = rigid_body_transformation([(residue['N9']['coord'],"N9"), (c4,"C4"), (residue['C1\'']['coord'],"C1'")], n_base)
        
        for key in arranged_n_base_coords:
            if key == "N9" or key == "C1'":
                continue
            residue[key] = {"coord":arranged_n_base_coords[key], "residue_idx":residue_idx}

    elif sequence[residue_idx] == 'U' or sequence[residue_idx] == 'C':  # pyrimidine
        direction_vector_c1_prime_2_n1 = get_direction(residue['C1\'']['coord'], residue['N1']['coord'])
        right_vector = np.cross(direction_vector_o4_prime_2_c1_prime, direction_vector_c1_prime_2_n1)
        right_vector = right_vector / np.linalg.norm(right_vector)

        # get C2 atom
        c2, _ = generate_atom(
            residue['N1']['coord'], bond_lengths_angles["bond_lengths"]["N1-C2"], bond_lengths_angles["bond_angles"]["C1'-N1-C2"], torsion_angles['chi'][residue_idx], direction_vector_c1_prime_2_n1, right_vector)
        
        arranged_n_base_coords = rigid_body_transformation([(residue['N1']['coord'],"N1"), (c2,"C2"), (residue['C1\'']['coord'],"C1'")], n_base)

        for key in arranged_n_base_coords:
            if key == "N1" or key == "C1'":
                continue
            residue[key] = {"coord":arranged_n_base_coords[key], "residue_idx":residue_idx}
       
    return residue, o3_prime


def post_process(structure):
    for i in range(len(structure)-1):
        next_p_coord = structure[i+1]['P']['coord']
        current_o3_prime_coord = structure[i]['O3\'']['coord']
        curret_p_coord = structure[i]['P']['coord']

        distance_o3_prime_2_next_p = np.linalg.norm(next_p_coord - current_o3_prime_coord)
        direction_o3_prime_2_next_p = get_direction(current_o3_prime_coord, next_p_coord)

        if distance_o3_prime_2_next_p > O3_prime_next_P_prime_distance:
            new_o3_prime_coord = current_o3_prime_coord + (distance_o3_prime_2_next_p-O3_prime_next_P_prime_distance) * direction_o3_prime_2_next_p

            direction_current_p_to_current_o3_prime = get_direction(curret_p_coord, current_o3_prime_coord)
            direction_current_p_to_new_o3_prime = get_direction(curret_p_coord, new_o3_prime_coord)
            angle_of_rotation = get_angle(direction_current_p_to_current_o3_prime, direction_current_p_to_new_o3_prime)
            rotation_axis = np.cross(direction_current_p_to_current_o3_prime, direction_current_p_to_new_o3_prime)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # ensuring unit vector

            structure[i] = rotate_rigid_body(structure[i], 'P', rotation_axis, angle_of_rotation)

    return structure
        

def build_rna(sequence, p_coords, torsion_angles):
    # structure = []
    prev_vector = get_direction(p_coords[0], p_coords[1])  # Initial reference vector

    y_axis = np.array([0, 1, 0])
    x_axis = np.array([1, 0, 0])

    rotate_y_to_prev_vector_matrix = get_rotation_matrix_vector(y_axis, prev_vector)
    prev_right_vector = np.dot(rotate_y_to_prev_vector_matrix, x_axis)

    structure = []
    residue = None
    current_res_o3_prime = None
    for i, nucleotide in enumerate(sequence):
        base_coord = p_coords[i]
        if i > 0:
            residue, current_res_o3_prime = build_residue_coords(base_coord, torsion_angles, prev_vector, prev_right_vector, sequence, i, residue) 
        else:
            residue, current_res_o3_prime = build_residue_coords(base_coord, torsion_angles, prev_vector, prev_right_vector, sequence, i)
        
        structure.append(residue)

        if i < len(sequence) - 1:
            next_base_coord = p_coords[i+1]
            next_vector = get_direction(current_res_o3_prime, next_base_coord)
            rotate_y_to_next_vector_matrix = get_rotation_matrix_vector(y_axis, next_vector)
            next_right_vector = np.dot(rotate_y_to_next_vector_matrix, x_axis)

            prev_vector = next_vector
            prev_right_vector = next_right_vector


    structure = post_process(structure)

    return structure

initialize()

