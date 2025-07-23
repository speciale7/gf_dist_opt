#10/01/25
#Giuseppe Speciale
import numpy as np
import os
import networkx as nx
import matplotlib.pyplot as plt
os.system('cls')
debug = 1
graphic = 0

def graph(NN, p_ER):
    I_NN = np.identity(NN, dtype=int)

    while True:
        G = nx.gnp_random_graph(NN, p_ER, directed=True)
        if nx.is_strongly_connected(G):
            break
    adj = nx.adjacency_matrix(G).todense()
    Adj = np.array(adj)
    pos = nx.spring_layout(G)  
    if graphic:
        nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", arrowsize=20)
        plt.title("Directed Strongly Connected Erdős-Rényi Graph")
        #plt.show()


    # Print the adjacency matrix
    Adj += I_NN 
    print("Adjacency Matrix:")
    print(Adj) 
    Adj_weighted = np.zeros((NN, NN))

    for ii in range(NN):
        out_neighbors = np.nonzero(Adj[ii])[0]  # Out-Neighbors of node ii
        deg_ii = len(out_neighbors)

        for jj in out_neighbors:
            deg_jj = len(np.nonzero(Adj[:, jj])[0])  # In-Neighbors of node jj
            Adj_weighted[ii, jj] = 1 / (1 + max(deg_ii, deg_jj))

    # Ensure double stochasticity
    for _ in range(100):  # Iterative normalization to achieve double stochasticity
        row_sums = np.sum(Adj_weighted, axis=1, keepdims=True)
        Adj_weighted /= row_sums  # Normalize rows
        col_sums = np.sum(Adj_weighted, axis=0, keepdims=True)
        Adj_weighted /= col_sums  # Normalize columns

    if debug:
        stoch = 0
        for ii in range(NN):
            stoch += row_sums[ii] + col_sums[0,ii]
        stoch /= 2*NN
        print(f"Double stochasticity check: {stoch} = 1.0")
        #print("Row sums:", row_sums)
        #print("Column sums:", col_sums)
        print("Weighted Adjacency Matrix:")
        print(Adj_weighted)

    return Adj, Adj_weighted
