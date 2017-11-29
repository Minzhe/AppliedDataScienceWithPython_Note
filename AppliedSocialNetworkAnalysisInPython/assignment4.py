import networkx as nx
import pandas as pd
import numpy as np
import pickle

P1_Graphs = pickle.load(open('A4_graphs','rb'))
P1_Graphs

def degree_dist(G):
    dgr = G.degree()
    dgr_values = sorted(set(dgr.values()))
    hist = [list(dgr.values()).count(x) / nx.number_of_nodes(G) for x in dgr_values]
    return hist
	
for G in P1_Graphs:
    print(nx.average_clustering(G), nx.average_shortest_path_length(G), len(degree_dist(G)))

def graph_identification():
    
    # Your Code Here
    method = list()
    for G in P1_Graphs:
        clustering_coef = nx.average_clustering(G)
        path_length = nx.average_shortest_path_length(G)
        dgr_hist = degree_dist(G)
        if len(dgr_hist) > 10:
            method.append('PA')
        elif clustering_coef < 0.1:
            method.append('SW_H')
        else:
            method.append('SW_L')
    
    return method # Your Answer Here


G = nx.read_gpickle('email_prediction.txt')

print(nx.info(G))

def salary_predictions():
    
    # Your Code Here
    node_attr = pd.DataFrame(index=G.nodes())
    node_attr['degree'] = pd.Series(nx.degree_centrality(G))
    node_attr['closeness'] = pd.Series(nx.closeness_centrality(G))
    node_attr['betweenness'] = pd.Series(nx.betweenness_centrality(G))
    node_attr['clustering'] = pd.Series(nx.clustering(G))
    node_attr['pagerank'] = pd.Series(nx.pagerank(G))
    node_attr['hits'] = pd.Series(nx.hits(G)[0])
    get_managment = lambda x: x[1]['ManagementSalary']
    node_attr['management'] = pd.Series([get_managment(node) for node in G.nodes(data=True)])
    
    X_train = node_attr.loc[node_attr.management.notnull()].iloc[:,:6]
    X_test = node_attr.loc[node_attr.management.isnull()].iloc[:,:6]
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = node_attr.management[node_attr.management.notnull()]

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    prob = pd.Series(clf.predict_proba(X_test)[:,1], index=node_attr.index[node_attr.management.isnull()])
    
    return prob # Your Answer Here



def new_connections_predictions():
    
    # Your Code Here
    get_edge_attr = lambda edges: {(edge[0],edge[1]):edge[2] for edge in edges}

    for node in G.nodes():
        G.node[node]['community'] = G.node[node]['Department']

    pref_attachment = get_edge_attr(list(nx.preferential_attachment(G)))
    edge_attr = pd.DataFrame(index=pref_attachment.keys())
    edge_attr['pref_attachment'] = pd.Series(pref_attachment)
    # edge_attr['common_neighbor'] = [len(list(nx.common_neighbors(G, edge[0], edge[1]))) for edge in edge_attr.index.values]
    edge_attr['jaccard'] = pd.Series(get_edge_attr(list(nx.jaccard_coefficient(G))))
    edge_attr['resource_allocation'] = pd.Series(get_edge_attr(list(nx.resource_allocation_index(G))))
    edge_attr['adamic_adar'] = pd.Series(get_edge_attr(list(nx.adamic_adar_index(G))))
    edge_attr['common_neighbor'] = pd.Series(get_edge_attr(list(nx.cn_soundarajan_hopcroft(G))))

    edge_attr = pd.merge(edge_attr, future_connections, left_index=True, right_index=True, how='outer')
    edge_attr.iloc[:,:5] = StandardScaler().fit_transform(edge_attr.iloc[:,:5])
    X_train = edge_attr.iloc[:,:5][edge_attr['Future Connection'].notnull()]
    y_train = edge_attr['Future Connection'][edge_attr['Future Connection'].notnull()]
    X_test = edge_attr.iloc[:,:5][edge_attr['Future Connection'].isnull()]

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:,1]
    predcition = pd.Series(y_pred, index=X_test.index)
    
    return predcition # Your Answer Here