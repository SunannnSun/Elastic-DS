import numpy as np

def gram_schmidt(vectors):
    """Perform Gram-Schmidt orthogonalization on a set of vectors."""
    orthogonal_vectors = []
    for v in vectors:
        w = v - sum(np.dot(v, u) * u for u in orthogonal_vectors)
        if np.linalg.norm(w) > 1e-10:  # Avoid division by zero
            orthogonal_vectors.append(w / np.linalg.norm(w))
    return np.array(orthogonal_vectors)

def tangent_plane_basis(P):
    """Compute an orthonormal basis for the tangent plane at point P on the 3-sphere."""
    dim = P.shape[0]
    P = P / np.linalg.norm(P)  # Ensure P is a unit vector
    # Start with standard basis vectors
    basis = np.eye(dim)
    # Apply Gram-Schmidt to get orthonormal vectors orthogonal to P
    basis = gram_schmidt([b - np.dot(b, P) * P for b in basis])
    return basis  # Return the 3 vectors orthogonal to P

def rotate_vector_in_tangent_plane(P, u, theta, phi, psi):
    """Rotate vector u in the tangent plane at point P using Euler angles theta, phi, psi."""
    # Find the tangent plane basis at P
    basis = tangent_plane_basis(P)
    
    # Express u in terms of the tangent plane basis
    coords = np.dot(u, basis.T)
    
    # Construct the rotation matrix (using ZYX Euler angles as an example)
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0, 1, 0],
        [-np.sin(phi), 0, np.cos(phi)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(psi), -np.sin(psi)],
        [0, np.sin(psi), np.cos(psi)]
    ])
    R = Rz @ Ry @ Rx
    
    # Rotate the coordinates
    rotated_coords = R @ coords
    
    # Convert back to the original coordinates
    rotated_vector = np.dot(rotated_coords, basis)
    return rotated_vector

# Example usage
P = np.array([2**0.5/2, 2**0.5/2]) 

basis = tangent_plane_basis(P)

# u = np.array([0, 1, 0, 0])  # Vector in the tangent plane at P
# theta = np.pi / 4  # Rotation angle around z-axis
# phi = np.pi / 6  # Rotation angle around y-axis
# psi = np.pi / 8  # Rotation angle around x-axis

# rotated_vector = rotate_vector_in_tangent_plane(P, u, theta, phi, psi)
# print("Rotated Vector:")
# print(rotated_vector)

a = 1