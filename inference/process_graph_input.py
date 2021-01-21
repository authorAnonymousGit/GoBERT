import torch
from torch_geometric.data import Data
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import create_input_dict
import numpy as np


class GraphInputReader:
    def __init__(self, sub_models, files_path, dataset_type, labels_num, sub_nn,
                 edge_creation_procedure='star', bidirectional=True, filter_by_origin_pred=None,
                 primary_model_name="primary"):
        self.input_dict, self.primary_df = create_input_dict(files_path,
                                                             dataset_type,
                                                             sub_models,
                                                             labels_num,
                                                             sub_nn,
                                                             filter_by_origin_pred=filter_by_origin_pred,
                                                             model_name=primary_model_name)
        # if filter_by_origin_pred:
        #     print('*'*20)
        #     print("Data Reader for Label ", filter_by_origin_pred)
        # print("Number of samples - ", dataset_type, ":")
        # print(len(self.input_dict))
        self.graph_idxs = list(self.input_dict.keys())
        self.labels_num = labels_num
        self.bidirectional = bidirectional
        self.primary_predictions, self.primary_dist = self.create_primary_predictions()
        self.models_to_nodes = self.map_models_to_nodes()
        self.nodes_to_models = self.map_nodes_to_models()
        self.graphs_node_embeddings = self.create_graphs_node_embeddings()
        self.graphs_edge_index = self.create_graphs_edge_index(edge_creation_procedure)
        self.graphs_label, self.graphs_label_encoding = self.create_graphs_label()
        # Add edge features
        self.graphs_edge_embeddings = self.create_graphs_edge_embeddings()
        self.dataset, self.data_idx_to_key_idx = self.create_dataset()
        self.primary_nodes = self.get_primary_nodes()
        self.num_of_nodes = self.get_num_of_nodes()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], self.data_idx_to_key_idx[index], \
               self.primary_predictions[index], self.primary_dist[index], \
               self.primary_nodes[index], self.num_of_nodes[index], \
               self.graphs_label_encoding[index]

    def map_models_to_nodes(self):
        # For each input instance, assign each sub model a node index
        models_to_nodes = dict()
        for key_idx in self.graph_idxs:
            filtered_nodes = {model_name for model_name, probs
                              in self.input_dict[key_idx].items()
                              if type(probs) == np.ndarray}
            models_to_nodes[key_idx] = {model_name: new_idx for new_idx, model_name
                                        in enumerate(filtered_nodes)}
        return models_to_nodes

    def map_nodes_to_models(self):
        nodes_to_models = dict()
        for key_idx in self.graph_idxs:
            nodes_to_models[key_idx] = dict([(value, key) for key, value
                                             in self.models_to_nodes[key_idx].items()])
        return nodes_to_models

    def create_graphs_node_embeddings(self):
        nodes_embeddings_dict = dict()
        for key_idx in self.graph_idxs:
            nodes_num = len(self.nodes_to_models[key_idx])
            graph_embeddings = []
            for node_idx in range(nodes_num):
                model_name = self.nodes_to_models[key_idx][node_idx]
                node_embedding = self.input_dict[key_idx][model_name]
                graph_embeddings.append(node_embedding)
            graph_embeddings = torch.tensor(graph_embeddings, dtype=torch.float)
            nodes_embeddings_dict[key_idx] = graph_embeddings
        return nodes_embeddings_dict

    def create_primary_predictions(self):
        primary_predictions = []
        dist = []
        for key_idx in self.graph_idxs:
            primary_predictions.append(self.primary_df.loc[self.primary_df['key_index']
                                                           == key_idx]['prediction'].item())
            dist.append(self.primary_df.loc[self.primary_df['key_index']
                                            == key_idx]['probability'].item())

        return primary_predictions, dist

    def create_graphs_edge_index(self, procedure):
        if procedure == 'star':
            return self.create_star_graph()
        elif procedure == 'complete':
            return self.create_complete_graph()
        elif procedure.startswith('pyramid_narrow') or procedure.startswith('pyramid_wide'):
            pr_to_dist = self.labels_num - 1 if procedure.startswith('pyramid_narrow') \
                else 1
            conn_sub_of_dist_1 = procedure.endswith('_conn_dist_1')
            return self.create_tree_graph(pr_to_dist=pr_to_dist, conn_dist_1=conn_sub_of_dist_1)

    def create_graphs_label(self):
        labels_dict = dict()
        labels_encoding_dict = list()
        for key_idx in self.graph_idxs:
            label = torch.tensor([self.primary_df.loc[
                                                     self.primary_df['key_index']
                                                     == key_idx]['label'].item() - 1])
            label_encoding = torch.nn.functional.one_hot(label.squeeze(0),
                                                         num_classes=self.labels_num)
            labels_dict[key_idx] = label
            labels_encoding_dict.append(label_encoding)
        return labels_dict, labels_encoding_dict

    def create_graphs_edge_embeddings(self):
        edges_embeddings_dict = dict()
        for key_idx in self.graph_idxs:
            graph_edges = self.graphs_edge_index[key_idx]
            graph_edge_embeddings = []
            for edge in graph_edges:
                model_name_out_edge = self.nodes_to_models[key_idx][edge[0].item()]
                rel_labels_model_1 = list(range(self.labels_num)) if model_name_out_edge == 'primary' \
                    else [int(model_name_out_edge[-2]) - 1, int(model_name_out_edge[-1]) - 1]

                model_name_in_edge = self.nodes_to_models[key_idx][edge[1].item()]
                rel_labels_model_2 = list(range(self.labels_num)) if model_name_in_edge == 'primary' \
                    else [int(model_name_in_edge[-2]) - 1, int(model_name_in_edge[-1]) - 1]

                common_labels = list(set(rel_labels_model_1) & set(rel_labels_model_2))
                edge_embedding = np.zeros(self.labels_num)
                edge_embedding[common_labels] = 1

                graph_edge_embeddings.append(edge_embedding)
            graph_edge_embeddings = torch.tensor(graph_edge_embeddings, dtype=torch.float)
            edges_embeddings_dict[key_idx] = graph_edge_embeddings
        return edges_embeddings_dict

    def create_dataset(self):
        # Data refers to object of type data.
        # This is a mapping from data objects list
        # to the key indexes of the original dataset
        data_idx_to_key_idx = dict()
        data_objects_list = list()
        for data_idx, key_idx in enumerate(self.graph_idxs):
            x = self.graphs_node_embeddings[key_idx]
            edge_index = self.graphs_edge_index[key_idx]
            edge_embeddings = self.graphs_edge_embeddings[key_idx]
            y = self.graphs_label[key_idx]
            data_object = Data(x=x, edge_index=edge_index,
                               edge_attr=edge_embeddings, y=y)
            data_objects_list.append(data_object)
            data_idx_to_key_idx[data_idx] = key_idx
        return data_objects_list, data_idx_to_key_idx

    def create_star_graph(self):
        # The primary node is connected to all the other nodes (sub NNs)
        # The sub NNs are not connected to each other
        edge_index_dict = dict()
        for key_idx in self.graph_idxs:
            primary_node_idx = self.models_to_nodes[key_idx]['primary']
            nodes_num = len(self.nodes_to_models[key_idx])
            edge_index = []
            for node_idx in range(nodes_num):
                if primary_node_idx == node_idx:
                    continue
                edge_index.append([node_idx, primary_node_idx])
                if self.bidirectional:
                    edge_index.append([primary_node_idx, node_idx])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_index_dict[key_idx] = edge_index
        return edge_index_dict

    def create_complete_graph(self):
        # Every pair of nodes is directly connected
        edge_index_dict = dict()
        for key_idx in self.graph_idxs:
            nodes_num = len(self.nodes_to_models[key_idx])
            edge_index = []
            for node_idx in range(nodes_num):
                for node_idx2 in range(nodes_num):
                    edge_index.append([node_idx, node_idx2])
                    edge_index.append([node_idx2, node_idx])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_index_dict[key_idx] = edge_index
        return edge_index_dict

    def create_tree_graph(self, pr_to_dist, conn_dist_1):
        # Currently, this function handles only cases where the entire set of possible sub models is available
        # pr_to_dist indicates whether to connect the primary network to model_15 (when pr_to_dist = 4)
        # or to the sub_nns with labels distance of 1 (when pr_to_dist = 1)
        edge_index_dict = dict()
        for key_idx in self.graph_idxs:
            primary_node_idx = self.models_to_nodes[key_idx]['primary']
            distances_dict = {dist: [] for dist in range(1, self.labels_num)}
            edge_index = []
            distances_idxs_dict = dict()
            for sub_model in self.models_to_nodes[key_idx].keys():
                if sub_model == 'primary':
                    continue
                dist = int(sub_model[-1]) - int(sub_model[-2])
                distances_dict[dist].append(sub_model)
            # print(distances_dict)
            # print(self.models_to_nodes[key_idx])
            for dist in range(1, self.labels_num):
                distances_dict[dist] = sorted(distances_dict[dist])
                # print(distances_dict[dist])
                distances_idxs_dict[dist] = [self.models_to_nodes[key_idx][sub_model]
                                             for sub_model in distances_dict[dist]]
                # print(distances_idxs_dict[dist])

                # Connect sub-nns to the primary network
                if pr_to_dist == dist:
                    edge_index.extend([[primary_node_idx, node_idx] for node_idx in distances_idxs_dict[dist]])
                    edge_index.extend([[node_idx, primary_node_idx] for node_idx in distances_idxs_dict[dist]])

                if dist == 1:
                    for node_pos in range(len(distances_idxs_dict[dist]) - 1):
                        node1 = distances_idxs_dict[dist][node_pos]
                        node2 = distances_idxs_dict[dist][node_pos + 1]
                        if conn_dist_1:
                            edge_index.extend([[node1, node2], [node2, node1]])
                else:
                    for node_pos, child in enumerate(distances_idxs_dict[dist]):
                        child = distances_idxs_dict[dist][node_pos]
                        parent1 = distances_idxs_dict[dist - 1][node_pos]
                        parent2 = distances_idxs_dict[dist - 1][node_pos + 1]
                        edge_index.extend([[child, parent1], [parent1, child],
                                           [child, parent2], [parent2, child]])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_index_dict[key_idx] = edge_index
        return edge_index_dict

    def get_primary_nodes(self):
        primary_nodes = []
        for key_idx in self.graph_idxs:
            primary_nodes.append(self.models_to_nodes[key_idx]['primary'])
        return primary_nodes

    def get_num_of_nodes(self):
        num_of_nodes = []
        for key_idx in self.graph_idxs:
            num_of_nodes.append(len(self.models_to_nodes[key_idx]))
        return num_of_nodes

