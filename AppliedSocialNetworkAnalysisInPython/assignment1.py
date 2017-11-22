import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms import bipartite


# This is the set of employees
employees = set(['Pablo',
                 'Lee',
                 'Georgia',
                 'Vincent',
                 'Andy',
                 'Frida',
                 'Joan',
                 'Claude'])

# This is the set of movies
movies = set(['The Shawshank Redemption',
              'Forrest Gump',
              'The Matrix',
              'Anaconda',
              'The Social Network',
              'The Godfather',
              'Monty Python and the Holy Grail',
              'Snakes on a Plane',
              'Kung Fu Panda',
              'The Dark Knight',
              'Mean Girls'])


# you can use the following function to plot graphs
# make sure to comment it out before submitting to the autograder
def plot_graph(G, weight_name=None):
    '''
    G: a networkx G
    weight_name: name of the attribute for plotting edge weights (if G is weighted)
    '''
    %matplotlib notebook
    import matplotlib.pyplot as plt
    
    plt.figure()
    pos = nx.spring_layout(G)
    edges = G.edges()
    weights = None
    
    if weight_name:
        weights = [int(G[u][v][weight_name]) for u,v in edges]
        labels = nx.get_edge_attributes(G,weight_name)
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
        nx.draw_networkx(G, pos, edges=edges, width=weights);
    else:
        nx.draw_networkx(G, pos, edges=edges);



def answer_one():
        
    movie_choices = nx.read_edgelist('Employee_Movie_Choices.txt', delimiter='\t')
    
    return movie_choices # Your Answer Here

    
def answer_two():
    
    # Your Code Here
    net = answer_one()
    for node in net.nodes():
        if node in employees:
            net.add_node(node, type='employee')
        elif node in movies:
            net.add_node(node, type='movie')
    
    return net # Your Answer Here


def answer_three():
        
    # Your Code Here
    net = answer_two()
    proj_net = bipartite.weighted_projected_graph(B=net, nodes=employees)
    
    return proj_net # Your Answer Here


def answer_four():
        
    # Your Code Here
    rel_net = nx.read_edgelist('Employee_Relationships.txt', delimiter='\t', data=[('relationship_score', int)])
    rel_df = pd.DataFrame(rel_net.edges(data=True), columns=['A', 'B', 'score'])
    rel_df.score = rel_df.score.apply(lambda x: x['relationship_score'])
    rel_dict = {tuple(sorted([rel_df.A[idx], rel_df.B[idx]])):rel_df.score[idx] for idx, row in rel_df.iterrows()}
    rel_df = pd.DataFrame(list(rel_dict.items()), columns=['pair', 'score'])

    proj_net = answer_three()
    proj_df = pd.DataFrame(proj_net.edges(data=True), columns=['A', 'B', 'common'])
    proj_df.common = proj_df.common.apply(lambda x: x['weight'])
    proj_dict = {tuple(sorted([proj_df.A[idx], proj_df.B[idx]])):proj_df.common[idx] for idx, row in proj_df.iterrows()}
    proj_df = pd.DataFrame(list(proj_dict.items()), columns=['pair', 'common'])
    rel_df = rel_df.merge(proj_df, how='outer', on = ['pair'])
    rel_df.common.fillna(0, inplace=True)
   
    return rel_df.score.corr(rel_df.common) # Your Answer Here