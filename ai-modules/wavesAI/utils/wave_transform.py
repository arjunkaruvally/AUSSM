## In this file, the functions cater to converting existing SSMs to wave canonical form
import numpy as np
import copy

def ssm_to_wavessm(U, V):
    """
    Convert an SSM to the wave canonical form
    :param U: numpy array of size n \times n where n is the dimensions of the RNN hidden state
    :param V: numpy array of size n \times d where d is the dimensions of the input vector
    :return: a tuple (B, B_inv) where B is the set of basis vectors for the wave subspace and B_inv is the basis dual
    """
    generator_set = [ V[:, i] for i in range(V.shape[1]) ]
    B = V.copy()

    i = 0
    while i < np.linalg.matrix_rank(U) and len(generator_set) > 0:
        generator_set_filt = [ True for j in range(len(generator_set)) ]
        for v_idx, v in enumerate(generator_set):
            Bcandidate = np.append(B, np.linalg.matrix_power(U, i+1) @ V[:, v_idx:v_idx+1], axis=1)
            if np.linalg.matrix_rank(Bcandidate) > np.linalg.matrix_rank(B):
                B = Bcandidate
            else:
                generator_set_filt[v_idx] = False
        generator_set = [ v for v_idx, v in enumerate(generator_set) if generator_set_filt[v_idx] ]
        i=i+1

    return B, np.linalg.pinv(B)


def ssm_to_wavessm_v1(U, V):
    """
    Convert an SSM to the wave canonical form
    :param U: numpy array of size n \times n where n is the dimensions of the RNN hidden state
    :param V: numpy array of size n \times d where d is the dimensions of the input vector
    :return: a tuple (B, B_inv) where B is the set of basis vectors for the wave subspace and B_inv is the basis dual
    """
    # Step 1 - take the columns of V in order
    B = None
    for b_id in range(V.shape[1]):
        v_gen = V[:, b_id:b_id+1]

        b_candidate = v_gen

        # 1.1 For each vector create the Krylov subspace
        while True:
            if B is None:
                B = b_candidate

                # generate the new krylov subspace element
                b_candidate = U @ b_candidate
                continue

            # try adding the candidate to the new basis
            B_new = np.append(B, b_candidate, axis=1)

            # 1.1 for each generated vector, check to see if linearly dependent (linear dependence check only
            # necessary for the first Krylov subspace elements)
            if np.linalg.matrix_rank(B_new) > np.linalg.matrix_rank(B):
                B = B_new
            else:
                # if the vector does not increase, its iterations cannot be expected to add to the dimensionality
                break

            # generate the new krylov subspace element
            b_candidate = U @ b_candidate

    return B, np.linalg.pinv(B)
