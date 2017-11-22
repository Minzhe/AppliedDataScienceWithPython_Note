import networkx as nx

G1 = nx.read_gml('friendships.gml')

def answer_one():
        
    # Your Code Here
    dgr = nx.degree_centrality(G1)[100]
    cls = nx.closeness_centrality(G1)[100]
    btn = nx.betweenness_centrality(G1, endpoints=False, normalized=True)[100]
    
    return (dgr, cls, btn) # Your Answer Here


def answer_two():
        
    # Your Code Here
    dgr = nx.degree_centrality(G1)
    
    return max(dgr.keys(), key=lambda x: dgr[x]) # Your Answer Here

def answer_three():
        
    # Your Code Here
    cls = nx.closeness_centrality(G1)
    
    return max(cls.keys(), key=lambda x: cls[x]) # Your Answer Here

def answer_four():
        
    # Your Code Here
    btn = nx.betweenness_centrality(G1)
    
    return max(btn.keys(), key=lambda x: btn[x]) # Your Answer Here

G2 = nx.read_gml('blogs.gml')

def answer_five():
        
    # Your Code Here
    pr = nx.pagerank(G2, alpha=0.85)
    
    return pr['realclearpolitics.com'] # Your Answer Here

def answer_six():
        
    # Your Code Here
    pr = nx.pagerank(G1, alpha=0.85)
    
    return sorted(pr.keys(), key=lambda x:pr[x], reverse=True)[:5] # Your Answer Here

def answer_seven():
        
    # Your Code Here
    hits = nx.hits(G2)
    
    return hits[0]['realclearpolitics.com'], hits[1]['realclearpolitics.com'] # Your Answer Here

def answer_eight():
        
    # Your Code Here
    hubs = nx.hits(G2)[0]
    
    return sorted(hubs.keys(), key=lambda x: hubs[x], reverse=True)[:5] # Your Answer Here

def answer_nine():
        
    # Your Code Here
    auts = nx.hits(G2)[1]
    
    return sorted(auts.keys(), key=lambda x: auts[x], reverse=True)[:5] # Your Answer Here