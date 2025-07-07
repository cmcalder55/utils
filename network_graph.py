import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def load_graph_data(nodes_path, edges_path):
    edgelist = pd.read_csv(edges_path, header=None).to_numpy().tolist()
    nodelist = pd.read_csv(nodes_path, header=0)
    return nodelist, edgelist

def preprocess_nodes(nodelist, ystart, q_cut, labels, TENURE_ATTR, SUBMITTED_ATTR):
    nodelist["tenure"] = ystart - nodelist["since"]
    nodelist = nodelist.iloc[:, [5, 6, 1]]  # Keep specific columns
    nodelist[TENURE_ATTR] = pd.qcut(nodelist["tenure"], q=q_cut, labels=labels)
    nodelist[SUBMITTED_ATTR] = pd.qcut(nodelist["submitted"], q=q_cut, labels=labels)
    nodelist = nodelist.set_index(nodelist.author)
    return nodelist

def build_graph(edgelist, nodelist, TENURE_ATTR, SUBMITTED_ATTR):
    G = nx.DiGraph()
    G.add_edges_from(edgelist)
    for i in G.nodes:
        G.add_nodes_from([i], tenure=nodelist.loc[i][TENURE_ATTR])
        G.add_nodes_from([i], submitted=nodelist.loc[i][SUBMITTED_ATTR])
    return G

def analyze_graph(G):
    neighbors = list(nx.neighbors(G, "DailyCollection"))
    adj_matrix = nx.incidence_matrix(G).todense()
    return neighbors, adj_matrix

def visualize_graph(G, nodelist, TENURE_ATTR, NODE_SIZE_FACTOR):
    node_size = [x * NODE_SIZE_FACTOR for x in nx.degree_centrality(G).values()]
    node_color = [nodelist.loc[i][TENURE_ATTR] for i in G.nodes]

    print(f"# of edges: {G.number_of_edges()}\n",
          f"# of nodes: {G.number_of_nodes()}\n",
          f"Directed Graph: {nx.is_directed(G)}\n",
          f"Density: {nx.density(G):.3}\n")

    plt.figure(figsize=(12, 10), dpi=100)
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx(G,
                     node_size=node_size,
                     pos=pos,
                     node_color=node_color,
                     arrowsize=20,
                     edge_color="grey",
                     with_labels=False)
    plt.title('Follower-Followee Network of Designers')
    plt.axis("off")
    plt.show()
