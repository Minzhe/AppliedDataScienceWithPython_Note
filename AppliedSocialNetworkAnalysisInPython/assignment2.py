import networkx as nx
import operator

# This line must be commented out when submitting to the autograder
# !head email_network.txt

def answer_one():
    
    # Your Code Here
    G = nx.read_edgelist('email_network.txt', delimiter='\t', data=[('time', int)], create_using=nx.MultiGraph())
    
    return G # Your Answer Here


def answer_two():
        
    # Your Code Here
    G = answer_one()
    
    return len(G.nodes()), len(G.edges()) # Your Answer Here


def answer_three():
        
    # Your Code Here
    G = answer_one()
    
    return nx.is_strongly_connected(G), nx.is_weakly_connected(G) # Your Answer Here


def answer_four():
        
    # Your Code Here
    G = answer_one()
    comps = nx.weakly_connected_components(G)
    
    return len(max(comps, key=len)) # Your Answer Here


def answer_five():
        
    # Your Code Here
    G = answer_one()
    comps = nx.strongly_connected_components(G)
    
    return len(max(comps, key=len)) # Your Answer Here


def answer_six():
        
    # Your Code Here
    G = answer_one()
    sub_G = nx.strongly_connected_component_subgraphs(G)
    G_sc = max(sub_G, key=len)
    
    return G_sc # Your Answer Here


def answer_seven():
        
    # Your Code Here
    G = answer_six()
    
    return nx.average_shortest_path_length(G) # Your Answer Here


def answer_eight():
        
    # Your Code Here
    G = answer_six()
    
    return nx.diameter(G) # Your Answer Here


def answer_nine():
       
    # Your Code Here
    G = answer_six()
    
    return set(nx.periphery(G)) # Your Answer Here


def answer_eleven():
        
    # Your Code Here
    G = answer_six()
    d = nx.diameter(G)
    periphery_nodes = nx.periphery(G)
    f = lambda x: [target for target, dist in nx.shortest_path_length(G, x).items() if dist == d]
    node_count = [(node, len(f(node))) for node in periphery_nodes]
    
    return max(node_count, key=operator.itemgetter(1)) # Your Answer Here


def answer_twelve():
        
    # Your Code Here
    G = answer_six()
    center = nx.center(G)[0]
    node = answer_eleven()[0]
    
    return len(nx.minimum_node_cut(G, center, node)) # Your Answer Here


def answer_thirteen():
        
    # Your Code Here
    G = answer_six()
    G_un = G.to_undirected(G)
    
    return G_un # Your Answer Here


def answer_fourteen():
        
    # Your Code Here
    G = answer_thirteen()
    
    return nx.transitivity(G), nx.average_clustering(G) # Your Answer Here