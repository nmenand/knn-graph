# knn-graph
A tool to create graphs from real vectors using K-nearest neighbors. 

## Requirements
- Python 3
- Numpy
- Scikit-learn

## Usage
```python3 knn-graph.py input_file.npy output_file k -s```

where:

```input_file.npy``` is a numpy array representing the n vectors

```output_file``` is the desired output file

```k``` is the number of nearest neighbors (the k in knn)

 ```-s``` Output to binary stream format

### Flags
```-s``` Output to binary stream format
```-r``` Randomize the stream (TODO)
```-d``` Add deletions to stream  (TODO)

### Example 
```python3 knn-graph.py sbm.npy sbm.graphstream 5 -s```
