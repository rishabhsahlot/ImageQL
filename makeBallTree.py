from sklearn.neighbors import BallTree
import pickle
import numpy as np
face_name = 'vlad_face_descriptors.p'
non_face_name = 'vlad_non_face_descriptors.p'

infile = open(face_name,'rb')
vlad_face_descriptors = pickle.load(infile)
infile.close()
leaf_size = 40
face_tree = BallTree(np.array(vlad_face_descriptors.values()),leaf_size)

with open('face_db', 'wb') as f:
	pickle.dump(face_tree, f,pickle.HIGHEST_PROTOCOL)
