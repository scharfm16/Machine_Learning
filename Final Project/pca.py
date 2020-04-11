import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from preprocessing import *



def pca_plot_train():
	X_train, y_train = get_preprocessed_data(write=True, datafile="stroke_train.csv")
	pca = PCA(n_components=2)
	pca.fit(X_train)
	X_pca = pca.fit_transform(X_train)
	plt.scatter(X_pca[:,0], X_pca[:,1], c = y_train.flatten())
	plt.xlabel('PC1')
	plt.ylabel('PC2')
	plt.show()

def pca_plot_all():
	X, y = get_preprocessed_data(write=True, datafile="stroke_all.csv")
	pca = PCA(n_components=2)
	pca.fit(X)
	X_pca = pca.fit_transform(X)
	plt.scatter(X_pca[:,0], X_pca[:,1], c =  y.flatten())
	plt.xlabel('PC1')
	plt.ylabel('PC2')
	plt.show()


