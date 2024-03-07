import pandas as pd
import numpy as np
import unittest

def construct_inputframe(phagerep, hostrep, interactions=None):
    """
    This function constructs the input dataframe for the machine learning model.

    INPUTS:
    - rbp_multirep: path to the multi_representations.csv file for the RBPs
    - loci_multirep: path to the multi_representations.csv file for the loci
    - path: path to the general data folder
    - data suffix to optionally add to the saved file name (default='')
    - interactions: optional path to the interactions.csv file if in training mode

    OUTPUTS: features, labels
    """
    features = []
    if interactions is not None:
        labels = []
        for i, host in enumerate(interactions['host']):
            phage = interactions['phage'][i]
            this_phagerep = np.asarray(phagerep.iloc[:, 1:][phagerep['accession'] == phage])
            this_hostrep = np.asarray(hostrep.iloc[:, 1:][hostrep['accession'] == host])
            features.append(np.concatenate([this_hostrep, this_phagerep], axis=1)) # first host rep, then phage rep!
            labels.append(interactions['interaction'][i])
    else:
        for i, host in enumerate(hostrep['accession']):
            for j, phage in enumerate(phagerep['accession']):
                features.append(pd.concat([hostrep.iloc[i, 1:], phagerep.iloc[j, 1:]], axis=1))
    features = np.vstack(features)
    return features

class TestConstructInputFrame(unittest.TestCase):
    def test_construct_inputframe(self):
        # Create example data
        phagerep = pd.DataFrame({'accession': [1, 2, 3, 4, 5],
                    'c1': [1, 1, 2, 3, 6],
                    'c2': [0, 1, 3, 9, 6],
                    'c3': [3, 5, 0, 7, 4],
                    'c4': [7, 0, 4, 8, 2],
                    'c5': [9, 1, 8, 5, 10]})
        hostrep = pd.DataFrame({'accession': ['b1', 'b2', 'b3'],
                'c1': [10, 7, 6],
                'c2': [7, 5, 2],
                'c3': [2, 10, 8],
                'c4': [2, 4, 6],
                'c5': [3, 9, 1]})    
        ints = pd.DataFrame({'host': ['b1', 'b2', 'b2', 'b3'], 'phage': [1, 2, 4, 3], 'interaction': [1, 1, 0, 1]})

        # Expected output
        expected_out = np.array([[10,  7,  2,  2,  3,  1,  0,  3,  7,  9],
                        [ 7,  5, 10,  4,  9,  1,  1,  5,  0,  1],
                        [ 7,  5, 10,  4,  9,  3,  9,  7,  8,  5],
                        [ 6,  2,  8,  6,  1,  2,  3,  0,  4,  8]])
        
        # Call the function
        out = construct_inputframe(phagerep, hostrep, interactions=ints)
        
        # Check if the output matches the expected output
        self.assertTrue(np.array_equal(out, expected_out))

if __name__ == '__main__':
    unittest.main()

