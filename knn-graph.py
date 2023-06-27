from sys import argv
import numpy as np
import scipy.sparse as sp
import struct
import random

from sklearn.neighbors import NearestNeighbors

import argparse

parser = argparse.ArgumentParser(description='Create graphs using K-nearest neighbors')
parser.add_argument("--output_stream", "-s", help="Output graph as binary stream", action="store_true")
parser.add_argument("--randomize_stream", "-r", help="Randomize the order of the binary stream", action="store_true")
parser.add_argument("data_path", help="path to directory containing dataset")
parser.add_argument("output_path", help="path to output file")
parser.add_argument("n_neighbors", help="number of nearest neighbors", type=int)
args = parser.parse_args()

k = args.n_neighbors
data_path = args.data_path
output_path = args.output_path
print('Loaded Args')

vecs = np.load(data_path)
print('Loaded Data')

neighbors = NearestNeighbors(n_neighbors=k, algorithm='auto')
neighbors.fit(vecs)
print('Fit Nearest Neighbors')

# Get K+1 nearest points for each vector in vecs.
nn_matrix = neighbors.kneighbors_graph(vecs, n_neighbors=k+1, mode='connectivity').astype(int).astype(bool)

# Make the matrix symmetric.
nn_graph = (nn_matrix + nn_matrix.transpose())

# Remove upper triangle & diagonal entries so all edges are represented once
graph = sp.tril(nn_graph, -1)
print('Generated KNN Graph')

if args.output_stream:
  num_nodes = vecs.shape[0]
  num_updates = graph.count_nonzero()
  
  # Using struct for binary formatting
  # See https://docs.python.org/3/library/struct.html for details
  output = open(output_path, "wb")
  output.write(struct.pack("=L", num_nodes))
  output.write(struct.pack("=Q", num_updates))

  for (x,y) in zip(*graph.nonzero()):
    output.write(struct.pack("=BLL", 1, x, y))
  output.close()

  print('Converted Graph to Stream')

else:
  np.savetxt(output_path +'.gz', vecs)
