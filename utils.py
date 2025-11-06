import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor
from sklearn.decomposition import PCA
import pymetis

import os

def load_data(file_name: str, feature_name: str, fold_num: int):
    # Always load from the dataset/ folder next to this file
    here = os.path.dirname(__file__)
    file_path = os.path.join(here, 'dataset', file_name)
    data = np.load(file_path, allow_pickle=True)
    feature = data[feature_name]
    adj_matrix = data['adj_matrix'].item().toarray()
    fold = data['fold']
    combos = [
        ([0, 1, 2], [3], [4]),
        ([1, 2, 3], [4], [0]),
        ([2, 3, 4], [0], [1]),
        ([3, 4, 0], [1], [2]),
        ([4, 0, 1], [2], [3]),
    ]
    combo = combos[fold_num]
    trainIndices, valIndices, testIndices = [], [], []
    for i, label in enumerate(fold):
        if label in combo[0]:
            trainIndices.append(i)
        elif label in combo[1]:
            valIndices.append(i)
        elif label in combo[2]:
            testIndices.append(i)
    trainX, valX, testX = feature[trainIndices], feature[valIndices], feature[testIndices]
    trainA = adj_matrix[trainIndices, :][:, trainIndices]
    valA = adj_matrix[valIndices, :][:, valIndices]
    testA = adj_matrix[testIndices, :][:, testIndices]
    trainA, valA, testA = (
        trainA + np.eye(trainA.shape[0]),
        valA + np.eye(valA.shape[0]),
        testA + np.eye(testA.shape[0]),
    )
    return trainX, trainA, valX, valA, testX, testA

def to_torch(X, adj_matrix, treat_binary, y, symmetric_norm=True, device="cpu"):
    X_torch = torch.tensor(X, dtype=torch.float32).to(device)
    adj_torch = torch.from_numpy(adj_matrix).to(torch.float32).to(device)
    degree = adj_torch.sum(dim=1)
    D_inv = torch.diag(torch.pow(degree, -1))
    if symmetric_norm:
      D_inv = torch.diag(torch.pow(degree, -1))
      A_norm = D_inv @ adj_torch
    else:
      D_inv_sqrt = torch.diag(torch.pow(degree, -0.5)).to(device)
      A_norm = D_inv_sqrt @ adj_torch @ D_inv_sqrt
    
    treat_torch = torch.tensor(treat_binary, dtype=torch.long).to(device)
    y_torch = torch.from_numpy(y).view(-1, 1).to(torch.float32).to(device)
    d_var=(1/degree).view(-1,1)
    return X_torch, A_norm, treat_torch, y_torch, d_var

def get_one_hop_neighbors(adj_matrix):
    one_hop_neighbors = []
    for i in range(adj_matrix.shape[0]):
      one_hop_neighbors.append(torch.tensor((np.where(adj_matrix[i,:]!=0)[0])))
    return one_hop_neighbors
def get_treat_neighbors(one_hop_neighbors, treat_binary):
    treatment=torch.tensor(np.where(treat_binary == 1)[0])
    treat_neighbor = [
        neighbor[torch.isin(neighbor, treatment)].to(dtype=torch.long)  # Include node i itself
        for i, neighbor in enumerate(one_hop_neighbors)
    ]
    return treat_neighbor

def plot_true_vs_pred(true_plot, pred_plot, title_name):
    # Convert PyTorch tensors to NumPy arrays if needed.
    if hasattr(true_plot, 'detach'):
        true_plot = true_plot.detach().cpu().numpy()
    if hasattr(pred_plot, 'detach'):
        pred_plot = pred_plot.detach().cpu().numpy()

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot of true vs. predicted values
    ax.scatter(true_plot, pred_plot, label='Data Points', alpha=0.7)

    # Generate a line y = x for comparison
    min_val = min(true_plot.min(), pred_plot.min())
    max_val = max(true_plot.max(), pred_plot.max())
    m = np.linspace(min_val, max_val, 100)
    ax.plot(m, m, color='red', label='y = x')

    # Set labels, title, legend, and grid
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title_name)
    ax.legend()
    ax.grid(True)

    plt.show()

def plot_PCA(X, ground_truth):

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    plt.figure(figsize=(8, 6))
    for cluster in np.unique(ground_truth):
        mask = (ground_truth == cluster)
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=colors[int(cluster)] if int(cluster) < len(colors) else None,
            label=f'Cluster {cluster}',
            alpha=0.5,
            edgecolors='k'
        )

    # Customize plot axes and title
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Visualization of Data")
    plt.legend(title="Clusters", fontsize=10, title_fontsize=12)
    
    # Display the plot
    plt.show()

def compute_effect(true_attention, adj_matrix):
    # Get the size of the matrix
    n = true_attention.shape[0]
    IME = torch.tensor(true_attention.diagonal(), dtype=torch.float32).numpy()  
    ISE = torch.tensor(np.sum(true_attention * (adj_matrix - np.eye(n)), axis=1), dtype=torch.float32).numpy()
    ITE = IME + ISE
    return IME, ISE, ITE

def plot_histogram(array, bins=10,title='histogram'):
    plt.hist(array.flatten(), bins=bins, edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()
def run_bagging(predictor,response,m_star,max_depth=15,n_estimators=500):
  estimator = DecisionTreeRegressor(max_depth=15, min_samples_split=2, random_state=42)

  # Create a BaggingRegressor ensemble using the estimator
  bagging_regressor = BaggingRegressor(estimator=estimator,
                                      n_estimators=n_estimators,  # adjust as needed
                                      random_state=42)
  # Fit the bagging regressor to the data
  bagging_regressor.fit(predictor, response)

  # Predict the response values using the ensemble
  response_bagging = bagging_regressor.predict(predictor)
  print(f"{np.sum((m_star - response_bagging) ** 2):4f}")
  return response_bagging
def run_boosting(predictor,response,m_star,max_depth=15,n_estimators=500):
  estimator = DecisionTreeRegressor(max_depth=15, min_samples_split=2, random_state=42)
  # Create an AdaBoost regressor ensemble using the estimator
  boosting_regressor = AdaBoostRegressor(estimator=estimator,
                                      n_estimators=n_estimators,  # adjust as needed
                                      random_state=42)
  boosting_regressor.fit(predictor, response)
  # Predict the response values using the ensemble
  response_boost = boosting_regressor.predict(predictor)
  print(f"{np.sum((m_star - response_boost) ** 2):4f}")
  return response_boost

def partition_with_metis(G, num_partitions):
    # METIS expects a list of adjacency lists
    adj = [list(G.neighbors(n)) for n in range(G.number_of_nodes())]
    (edgecuts, parts) = pymetis.part_graph(num_partitions,adj)
    return parts

def compute_metric(IME, ISE, ITE, IME_est, ISE_est, ITE_est, printing=True):
    # Compute AME (Absolute Mean Error)
    IME_AME = np.abs(np.mean(IME) - np.mean(IME_est))
    ISE_AME = np.abs(np.mean(ISE) - np.mean(ISE_est))
    ITE_AME = np.abs(np.mean(ITE) - np.mean(ITE_est))

    # Also report mean differences (signed)
    IME_AME = np.mean(IME) - np.mean(IME_est)
    ISE_AME = np.mean(ISE) - np.mean(ISE_est)
    ITE_AME = np.mean(ITE) - np.mean(ITE_est)

    # Compute PEHE (sqrt of mean squared error)
    IME_PEHE = np.sqrt(np.mean((IME - IME_est)**2))
    ISE_PEHE = np.sqrt(np.mean((ISE - ISE_est)**2))
    ITE_PEHE = np.sqrt(np.mean((ITE - ITE_est)**2))

    if printing:
        print("IME: AME = {:.5f}, PEHE = {:.5f}".format(IME_AME, IME_PEHE))
        print("ISE: AME = {:.5f}, PEHE = {:.5f}".format(ISE_AME, ISE_PEHE))
        print("ITE: AME = {:.5f}, PEHE = {:.5f}".format(ITE_AME, ITE_PEHE))

    results = {
        "IME_AME": round(float(IME_AME), 5),
        "ISE_AME": round(float(ISE_AME), 5),
        "ITE_AME": round(float(ITE_AME), 5),
        "IME_PEHE": round(float(IME_PEHE), 5),
        "ISE_PEHE": round(float(ISE_PEHE), 5),
        "ITE_PEHE": round(float(ITE_PEHE), 5)
    }
    return results

def Baseline_compute(y,mhat,ehat,adj_matrix,n,treat_binary):
    first_column=(treat_binary-ehat)
    # first_column=(treat_binary-e_star)
    second_column=(adj_matrix-np.eye(n)).dot(first_column)
    predictor=np.column_stack((first_column,second_column))/np.sum(adj_matrix,1).reshape(-1, 1)
    response = y - mhat
    # predictor=np.column_stack((first_column,second_column))
    beta_hat=np.linalg.inv(predictor.T @ predictor) @ (predictor.T @ response)
    # DBML method (two parameter estimate)
    IME_est=beta_hat[0]/np.sum(adj_matrix,1)
    ISE_est=np.sum(beta_hat[1]*(adj_matrix-np.eye(n)),1)/np.sum(adj_matrix,1)
    ITE_est=IME_est+ISE_est
    return IME_est,ISE_est,ITE_est,beta_hat