import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np


############################################################################################################### node/edge embedding
class GraphEmbeddings(nn.Module):
    def __init__(self, opt):
        super(GraphEmbeddings, self).__init__()
        self.embed_dim = opt.embed_dim
        self.embed_freeze = opt.embed_freeze
 
        self.device = opt.device if hasattr(opt, 'device') else 'cpu'

        # Define labels for nodes and edges
        self.node_labels = self.get_node_labels()
        self.edge_labels = self.get_edge_labels()

        self.num_node_labels = len(self.node_labels)
        self.num_edge_labels = len(self.edge_labels)
    
        assert(self.embed_dim==768)  # GPT-2 embeddings have 768 dimensions
        node_embeddings_path = os.path.join(opt.gpt2_path, 'node_embeddings.npy')
        edge_embeddings_path = os.path.join(opt.gpt2_path, 'edge_embeddings.npy')

        self.gpt2_mode = opt.gpt2_node_mode  # Mode for selecting node embedding type
        self.load_gpt2_embeddings(node_embeddings_path, edge_embeddings_path)
        self.node_embedding = nn.Embedding.from_pretrained(self.node_embedding_matrix, freeze=self.embed_freeze)
        self.edge_embedding = nn.Embedding.from_pretrained(self.edge_embedding_matrix, freeze=self.embed_freeze)


    def get_node_labels(self):
        # Define your node labels here
        # first  list: categories of nodes in sg data 
        # second list: missing cateories of the OPA dataset in sg data      
        node_labels = [ '__background__', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 
                        'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 
                        'boy', 'branch', 'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 
                        'clock', 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer',
                        'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food',
                        'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat',
                        'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 
                        'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain',
                        'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people', 'person', 
                        'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot', 
                        'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 
                        'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow', 
                        'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 
                        'towel', 'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable',
                        'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra'] +\
                        ["apple", "bicycle", "broccoli", "cake", "cell phone", "donut", "fire hydrant", "keyboard", 
                         "knife", "mouse", "potted plant","remote" ,"sandwich", "scissors", "spoon", "suitcase", "toothbrush", "wine glass"]

        return node_labels

    def get_edge_labels(self):
        # Define your edge labels here
        edge_labels = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                       'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for', 
                       'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                       'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over', 
                       'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on', 
                       'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']
        return edge_labels

    def forward_node(self, x):
        return self.node_embedding(x)

    def forward_edge(self, x):
        return self.edge_embedding(x)

    def load_gpt2_embeddings(self, node_embeddings_path, edge_embeddings_path):
            node_embeddings = np.load(node_embeddings_path, allow_pickle=True).item()
            edge_embeddings = np.load(edge_embeddings_path, allow_pickle=True).item()

            self.node_embedding_matrix = torch.zeros((self.num_node_labels, self.embed_dim), dtype=torch.float32)
            for idx, label in enumerate(self.node_labels):
                if label in node_embeddings:
                    self.node_embedding_matrix[idx] = torch.from_numpy(node_embeddings[label][self.gpt2_mode])
                else:
                    self.node_embedding_matrix[idx] = torch.zeros(self.embed_dim)

            self.edge_embedding_matrix = torch.zeros((self.num_edge_labels, self.embed_dim), dtype=torch.float32)
            for idx, label in enumerate(self.edge_labels):
                if label in edge_embeddings:
                    self.edge_embedding_matrix[idx] = torch.from_numpy(edge_embeddings[label])
                else:
                    self.edge_embedding_matrix[idx] = torch.zeros(self.embed_dim)

            self.node_embedding_matrix = self.node_embedding_matrix.to(self.device)
            self.edge_embedding_matrix = self.edge_embedding_matrix.to(self.device)


############################################################################################################### GTN (Graph Transformer Network)
class GTN_layer(nn.Module):
    def __init__(self, opt, input_dim, output_dim, dropout=0.1):
        super(GTN_layer, self).__init__()
        assert opt.gtn_output_dim % opt.gtn_num_head == 0, "Output dimension must be divisible by num_heads"
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = opt.gtn_num_head
        self.head_dim = self.output_dim // self.num_heads

        # Linear transformations for queries, keys, and values
        self.query_proj = nn.Linear(input_dim, output_dim)
        self.key_proj   = nn.Linear(input_dim, output_dim)
        self.value_proj = nn.Linear(input_dim, output_dim)

        # Edge feature transformation
        self.edge_proj = nn.Linear(input_dim, output_dim)

        # Output projection for node features
        self.out_proj = nn.Linear(output_dim, output_dim)

        # Projection for matching input and output dimensions
        self.input_proj = nn.Linear(input_dim, output_dim)

        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node feature matrix of shape (num_nodes, input_dim)
            edge_index: Edge indices in COO format, shape (2, num_edges)
            edge_attr: Edge feature matrix, shape (num_edges, input_dim)
        Returns:
            Updated node features of shape (num_nodes, output_dim)
            Updated edge features of shape (num_edges, output_dim)
        """
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        device = x.device

        # Compute queries, keys, and values for nodes
        Q = self.query_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        K = self.key_proj(x)  .view(num_nodes, self.num_heads, self.head_dim)
        V = self.value_proj(x).view(num_nodes, self.num_heads, self.head_dim)

        # Edge features projection
        E = self.edge_proj(edge_attr).view(num_edges, self.num_heads, self.head_dim)

        # Initialize attention scores with very negative values
        attn_scores = torch.full((num_nodes, num_nodes, self.num_heads), float('-inf'), device=device)

        # Compute attention scores for edges
        src_nodes = edge_index[0]  # Source nodes
        tgt_nodes = edge_index[1]  # Target nodes

        # Compute attention scores using edge features
        # Q_src: (num_edges, num_heads, head_dim)
        Q_src = Q[src_nodes]
        K_tgt = K[tgt_nodes]
        edge_features = E

        # Compute edge-aware attention scores
        # e_{ij} = (Q_i * K_j^T) + (Q_i * E_{ij}^T) + (E_{ij} * K_j^T)
        attn_edge = (Q_src * K_tgt).sum(dim=2)  # (num_edges, num_heads)
        attn_edge += (Q_src * edge_features).sum(dim=2)
        attn_edge += (edge_features * K_tgt).sum(dim=2)
        attn_edge = attn_edge / (self.head_dim ** 0.5)

        # Assign computed attention scores to attn_scores tensor
        attn_scores[src_nodes, tgt_nodes] = attn_edge

        # Self-loops
        attn_self = (Q * K).sum(dim=2) / (self.head_dim ** 0.5)
        
        attn_scores[torch.arange(num_nodes), torch.arange(num_nodes)] = attn_self 

        # Apply softmax to compute attention weights
        attn_weights = F.softmax(attn_scores, dim=1)  # (num_nodes, num_nodes, num_heads)
        attn_weights = self.dropout(attn_weights)

        # Compute updated node features
        # V: (num_nodes, num_heads, head_dim)
        node_out = torch.einsum('ijh,jhd->ihd', attn_weights, V)

        # Combine heads
        node_out = node_out.contiguous().view(num_nodes, -1)

        # Output projection for node features and residual connection
        node_out = self.out_proj(node_out)

        # Match dimensions of input and output for residual connection
        x_proj = self.input_proj(x)  # Project input x to output_dim
        
        # Apply Layer Normalization and residual connection
        node_out = self.layer_norm(x_proj + node_out)

        # For edge features, apply a simple transformation (can be customized)
        edge_out = self.edge_proj(edge_attr)  # Apply transformation to edge features
        
        return node_out, edge_out
###########################################################################
class GTN(nn.Module):
    def __init__(self, opt, dropout=0.1):
        """
        Args:
            opt: Options/configuration object (must include embed_dim, gtn_output_dim, gtn_hidden_dim, gtn_num_head, gtn_layer_num).
            dropout: Dropout probability to use in each GTN layer.
        """
        super(GTN, self).__init__()

        self.layers = nn.ModuleList()
        num_layers = opt.gtn_num_layer

        if num_layers >= 1:
            # First layer: input_dim = embed_dim, output_dim = gtn_hidden_dim (if more than one layer)
            output_dim = opt.gtn_hidden_dim if num_layers > 1 else opt.gtn_output_dim
            self.layers.append(GTN_layer(opt, opt.embed_dim, output_dim, dropout))

            # Intermediate layers: input_dim = gtn_hidden_dim, output_dim = gtn_hidden_dim
            for _ in range(num_layers - 2):
                self.layers.append(GTN_layer(opt, opt.gtn_hidden_dim, opt.gtn_hidden_dim, dropout))

            # Last layer: input_dim = gtn_hidden_dim, output_dim = gtn_output_dim (if more than one layer)
            if num_layers > 1:
                self.layers.append(GTN_layer(opt, opt.gtn_hidden_dim, opt.gtn_output_dim, dropout))

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node feature matrix of shape (num_nodes, input_dim).
            edge_index: Edge indices in COO format, shape (2, num_edges).
            edge_attr: Edge feature matrix, shape (num_edges, input_dim).
        Returns:
            Final updated node features and edge features.
        """
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
        return x, edge_attr



    def __init__(self, opt, dropout=0.1):
        super(GTN, self).__init__()

        self.layers = nn.ModuleList()
        num_layers = opt.gtn_num_layer

        if num_layers >= 1:
            output_dim = opt.gtn_hidden_dim if num_layers > 1 else opt.gtn_output_dim
            self.layers.append(GTN_layer(opt, opt.embed_dim, output_dim, dropout))

            for _ in range(num_layers - 2):
                self.layers.append(GTN_layer(opt, opt.gtn_hidden_dim, opt.gtn_hidden_dim, dropout))

            if num_layers > 1:
                self.layers.append(GTN_layer(opt, opt.gtn_hidden_dim, opt.gtn_output_dim, dropout))

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights with scene graph-specific considerations"""
        for layer in self.layers:
            # Query and Key projections: Use Kaiming initialization
            # This helps with the attention mechanism's gradient flow
            for module in [layer.query_proj, layer.key_proj]:
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    bound = 1 / math.sqrt(module.bias.shape[0])
                    nn.init.uniform_(module.bias, -bound, bound)

            # Value projection: Using xavier_normal_ for better value propagation
            nn.init.xavier_normal_(layer.value_proj.weight)
            if layer.value_proj.bias is not None:
                nn.init.zeros_(layer.value_proj.bias)

            # Edge projections: Important for relationship features
            # Using xavier_uniform_ with a smaller gain for more stable initial predictions
            nn.init.xavier_uniform_(layer.edge_proj.weight, gain=0.5)
            if layer.edge_proj.bias is not None:
                nn.init.zeros_(layer.edge_proj.bias)

            # Output and input projections
            for module in [layer.out_proj, layer.input_proj]:
                # Using xavier_uniform_ with custom gain for better stability
                nn.init.xavier_uniform_(module.weight, gain=0.8)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            
            # Layer norm: Initialize with slightly positive bias 
            # This helps prevent dead neurons in deep networks
            if hasattr(layer, 'layer_norm'):
                nn.init.ones_(layer.layer_norm.weight)
                nn.init.constant_(layer.layer_norm.bias, 0.1)

    def reset_parameters(self):
        """Reset all parameters by re-initializing them"""
        self.init_weights()

    def forward(self, x, edge_index, edge_attr):
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
        return x, edge_attr