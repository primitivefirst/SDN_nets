"""Open topologies for the experiments: NSFNET-like and fat-tree."""
import networkx as nx


def nsfnet():
    """NSFNET-like topology: 14 nodes, 21 bidirectional links."""
    edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5),
        (3, 4), (3, 6), (4, 5), (4, 13), (5, 7), (5, 12),
        (6, 8), (7, 8), (7, 9), (8, 9), (8, 11), (9, 10),
        (10, 11), (10, 13), (11, 13),
    ]
    G = nx.Graph()
    G.add_nodes_from(range(14))
    G.add_edges_from(edges)
    assert nx.is_connected(G), "NSFNET graph must be connected"
    assert G.number_of_edges() == 21, "NSFNET must have 21 links"
    return G, list(range(14))


def fat_tree(k=4):
    """Fat-tree topology (default k=4): 16 hosts + 20 switches, 48 links.

    Returns (graph, hosts): only hosts are used as traffic endpoints.
    """
    assert k % 2 == 0, "k must be even"
    pods = k
    agg_per = edge_per = k // 2
    hosts_per_edge = k // 2
    G = nx.Graph()
    nid = 0
    host_id, edge_id, agg_id, core_id = {}, {}, {}, {}
    hosts = []
    for p in range(pods):
        for e in range(edge_per):
            for h in range(hosts_per_edge):
                host_id[(p, e, h)] = nid
                hosts.append(nid)
                nid += 1
    for p in range(pods):
        for e in range(edge_per):
            edge_id[(p, e)] = nid
            nid += 1
        for a in range(agg_per):
            agg_id[(p, a)] = nid
            nid += 1
    for a in range(agg_per):
        for p2 in range(pods):
            core_id[(a, p2)] = nid
            nid += 1
    for p in range(pods):
        for e in range(edge_per):
            for h in range(hosts_per_edge):
                G.add_edge(host_id[(p, e, h)], edge_id[(p, e)])
            for a in range(agg_per):
                G.add_edge(edge_id[(p, e)], agg_id[(p, a)])
    for p in range(pods):
        for a in range(agg_per):
            for p2 in range(pods):
                G.add_edge(agg_id[(p, a)], core_id[(a, p2)])
    assert nx.is_connected(G), "fat-tree graph must be connected"
    return G, hosts


def get_topology(name):
    if name == "nsfnet":
        return nsfnet()
    if name == "fattree":
        return fat_tree(k=4)
    raise ValueError(f"Unknown topology: {name}")
