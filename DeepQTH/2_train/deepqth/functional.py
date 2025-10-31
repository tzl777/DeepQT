from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, List
from torch.multiprocessing import spawn
import torch
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt
import os
from scipy.spatial import Voronoi, ConvexHull
from collections import Counter

def dijkstra_source_to_all(G, source, max_path_length):
    if source not in G:
        raise nx.NodeNotFound("Source {} not in G".format(source))

    num_nodes = G.number_of_nodes()
    node_index = {node: i for i, node in enumerate(G.nodes())}
    short_path = torch.full((num_nodes, max_path_length), fill_value=-1)

    for target in G.nodes():
        if source == target:
            continue
        try:
            path = nx.shortest_path(G, source=source, target=target) 
            if len(path) <= max_path_length:
               
                padded_path = path + [-1] * (max_path_length - len(path))
                short_path[node_index[target], :] = torch.tensor(padded_path[:max_path_length])
        except nx.NetworkXNoPath:
            pass
    # print(short_path)
    return short_path



# def floyd_warshall_source_to_all(G, source, cutoff=None):
#     if source not in G:
#         raise nx.NodeNotFound("Source {} not in G".format(source))

#     edges = {edge: i for i, edge in enumerate(G.edges())} 

#     level = 0  # the current level
#     nextlevel = {source: 1}  # list of nodes to check at next level
#     node_paths = {source: [source]}  # paths dictionary  (paths to key from source)
#     edge_paths = {source: []}

#     while nextlevel:
#         thislevel = nextlevel
#         nextlevel = {}
#         for v in thislevel: 
#             for w in G[v]: 
#                 if w not in node_paths: 
#                     node_paths[w] = node_paths[v] + [w]
#                     edge_paths[w] = edge_paths[v] + [edges[tuple(node_paths[w][-2:])]]
#                     nextlevel[w] = 1 

#         level = level + 1

#         if (cutoff is not None and cutoff <= level):
#             break

#     return node_paths, edge_paths

def all_pairs_shortest_path(G, max_path_length) -> torch.Tensor:
    num_nodes = G.number_of_nodes()
    short_paths = torch.full((num_nodes, num_nodes, max_path_length), fill_value=-1)
    for n in G:
        short_paths[n] = dijkstra_source_to_all(G, n, max_path_length) 
    return short_paths


# def all_pairs_shortest_path(G) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
#     paths = {n: floyd_warshall_source_to_all(G, n) for n in G}
#     node_paths = {n: paths[n][0] for n in paths}
#     edge_paths = {n: paths[n][1] for n in paths}
#     return node_paths, edge_paths

def shortest_path_distance(graph, max_path_length):

    num_nodes = graph.number_of_nodes()
    batch =1
    paths = [all_pairs_shortest_path(graph, max_path_length)]
    node_paths = torch.full((batch, num_nodes, num_nodes, max_path_length), fill_value=-1) 
    edge_paths = torch.full((batch, num_nodes * num_nodes, max_path_length-1), fill_value=-1) 

    for i, path in enumerate(paths):
        node_paths[i] = path #[72,72,5]
        for j in range(path.shape[-1] - 1):  
            target = torch.tensor([list(path[:,:,j].flatten()), list(path[:,:,j+1].flatten())])

            target_start = target[0].unsqueeze(1)  # Shape: (5184, 1)
            target_end = target[1].unsqueeze(1)  # Shape: (5184, 1)

            edges = list(graph.edges)
            src_nodes = torch.tensor([edge[0] for edge in edges])
            dst_nodes = torch.tensor([edge[1] for edge in edges])

            matches = (target_start == src_nodes) & (target_end == dst_nodes)  # Shape: (5184, 2664)
            # print(matches.shape)

            matching_indices = matches.nonzero(as_tuple=False)
            # print(matching_indices.shape) #torch.Size([5112, 2])，

            edge_path = torch.full((target.size(1),), fill_value=-1, dtype=torch.long)

            if matching_indices.numel() > 0:
                edge_path[matching_indices[:, 0]] = matching_indices[:, 1]
            # print(edge_path.shape)
            edge_paths[i, :, j] = edge_path
    # print(edge_paths)
    edge_paths = edge_paths.reshape((batch, num_nodes, num_nodes, -1)) 

    return node_paths, edge_paths


def batched_shortest_path_distance(data, max_path_length):
    batch = len(data)
    graphs = [to_networkx(sub_data) for sub_data in data] 
    relabeled_graphs = []
    # shift = 0
    num_nodes = len(data[0].x)
    
    paths = [all_pairs_shortest_path(G, max_path_length) for G in graphs] 
    node_paths = torch.full((batch, num_nodes, num_nodes, max_path_length), fill_value=-1) #[3,72,72,5]
    edge_paths = torch.full((batch, num_nodes * num_nodes, max_path_length-1), fill_value=-1)

    for i, path in enumerate(paths):
        node_paths[i] = path #[72,72,5]
        node_mask = (path != -1)
        for j in range(path.shape[-1] - 1):  
            target = torch.tensor([list(path[:,:,j].flatten()), list(path[:,:,j+1].flatten())])
            # print(target)
            # print(data[i].edge_index.shape)

            target_start = target[0].unsqueeze(1)  # Shape: (5184, 1)
            target_end = target[1].unsqueeze(1)  # Shape: (5184, 1)

            matches = (target_start == data[i].edge_index[0]) & (target_end == data[i].edge_index[1])  # Shape: (5184, 2664)
            matching_indices = matches.nonzero(as_tuple=False)
           
            edge_path = torch.full((target.size(1),), fill_value=-1, dtype=torch.long)

            if matching_indices.numel() > 0: 
                edge_path[matching_indices[:, 0]] = matching_indices[:, 1]
            # print(edge_path.shape)
            edge_paths[i, :, j] = edge_path
    # print(edge_paths)
    edge_paths = edge_paths.reshape((batch, num_nodes, num_nodes, -1))  # torch.Size([3, 72, 72, 4])

    return node_paths, edge_paths

def split_batch_data(atom_fea, edge_idx, edge_fea, cart_coords, lattices, batch):
    data_list = []
    for graph_id in batch.unique():

        node_mask = (batch == graph_id)
        x = atom_fea[node_mask]
        cart_coord = cart_coords[node_mask]
        lattice = lattices[3*graph_id:3*(graph_id+1),:]
        # print(lattice)

        edge_mask = (batch[edge_idx[0]] == graph_id) & (batch[edge_idx[1]] == graph_id)
        edge_index = edge_idx[:, edge_mask]
        edge_attr = edge_fea[edge_mask]

        edge_index = edge_index - edge_index.min()

        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, cart_coord=cart_coord, lattice=lattice)
        data_list.append(graph_data)
    return data_list

def create_ptr_from_batch(batch): 
    node_counts = torch.bincount(batch)
    ptr = torch.zeros(node_counts.size(0) + 1, dtype=torch.long)
    ptr[1:] = torch.cumsum(node_counts, dim=0)
    return ptr

def extend_coords_periodic(coords, box_sizes, dim):
    extended_coords = []
    shifts = [-1, 0, 1] 
    if dim == 2:
        box_size = box_sizes[:2, :]

        for shift_x in shifts:
            for shift_y in shifts:
                shift_vector = np.array([shift_x, shift_y]).reshape(1,2) @ box_size
                extended_coords.append(coords + shift_vector)
    elif dim ==3:
        for shift_x in shifts:
            for shift_y in shifts:
                for shift_z in shifts:
                    shift_vector = np.array([shift_x, shift_y, shift_z]) @ box_sizes
                    extended_coords.append(coords + shift_vector)
    else:
        raise ValueError('Unknown dimension: {}'.format(dim))
    return np.vstack(extended_coords)


# 计算 Voronoi 多边形面积
def calculate_voronoi_areas(extended_coords, indices):
    vor = Voronoi(extended_coords)
    areas = []

    for region_index in vor.point_region[indices]:
        region = vor.regions[region_index]
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[i] for i in region]
            if len(polygon) > 0:
                hull = ConvexHull(polygon)
                area = hull.area
                areas.append(area)
            else:
                areas.append(0)
        else:
            areas.append(0)
    areas = [round(area, 4) for area in areas]
    return np.array(areas)

def calculate_voronoi_volumes(extended_coords, indices):
    vor = Voronoi(extended_coords)
    volumes = []
    for region_index in vor.point_region[indices]:
        region = vor.regions[region_index]
        if not -1 in region and len(region) > 0:
            polygon = [vor.vertices[j] for j in region]
            if len(polygon) > 0:
                hull = ConvexHull(polygon)
                volume = hull.volume
                volumes.append(volume)
            else:
                volumes.append(0)
        else:
            volumes.append(0)
    rounded_volumes = [round(volume, 4) for volume in volumes]
    return np.array(rounded_volumes)

def create_graph(atoms, attr_value):
    G = nx.Graph()
    for i, (x, y, z) in enumerate(atoms):
        G.add_node(i, pos=(x, y, z), attr=attr_value[i])
    
    threshold = 1.7
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            xi, yi, zi = atoms[i]
            xj, yj, zj = atoms[j]
            dist = ((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2) ** 0.5
            if dist < threshold:
                G.add_edge(i, j)
    return G

def cal_voronoi_and_centrality(cart_coords, lattice, dim):

    voronoi_values = []
    centralities = []

    atom_coord = np.array(cart_coords) 
    lattice = np.array(lattice)

    extended_coords = extend_coords_periodic(atom_coord, lattice, dim)

    a_dict = {tuple(elem): idx for idx, elem in enumerate(atom_coord)}
    b_dict = {tuple(elem): idx for idx, elem in enumerate(extended_coords)}

    indices = [b_dict[tuple(elem)] for elem in b_dict if elem in a_dict]
    if dim == 2:
        areas_or_volumes = torch.tensor(calculate_voronoi_areas(extended_coords[:, :2], indices),
                                        dtype=torch.float64)
    elif dim == 3:
        areas_or_volumes = torch.tensor(calculate_voronoi_volumes(extended_coords, indices), dtype=torch.float64)
    else:
        raise ValueError('Unknown dimension: {}'.format(dim))
    voronoi_values.append(areas_or_volumes)
    G = create_graph(atom_coord, areas_or_volumes)
    # visualize_2d_graph(G)
    centrality = torch.tensor(list(nx.laplacian_centrality(G).values()), dtype=torch.float64)
    centralities.append(centrality)
    # centrality = torch.tensor(list(nx.pagerank(G, weight=edge_attr).values())).view(-1, 1)
    # centrality = torch.tensor(list(nx.eigenvector_centrality(G, weight=edge_attr).values())).view(-1, 1)
    voronoi_values = torch.stack(voronoi_values).flatten().view(-1, 1)
    centralities = torch.stack(centralities).flatten().view(-1, 1)
    return voronoi_values, centralities

def visualize_2d_graph(G):
    plt.figure(figsize=(8, 2.7))
    pos = nx.get_node_attributes(G, 'pos')
    attrs = nx.get_node_attributes(G, 'attr')
    nodes = G.nodes()
    
    pos_2d = {node: coords[:2] for node, coords in pos.items()}

    node_colors = [attrs[node] for node in nodes] 
    cmap = plt.cm.viridis  

    nx.draw(G, pos_2d, with_labels=True, node_size=180, node_color=node_colors, edgecolors="tab:gray", font_size=6, font_weight='bold', width=3, cmap=cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])
    plt.colorbar(sm, label='Voronoi Values')
    plt.show()

