from scipy.sparse import csr_matrix
import numpy as np
import torch
import torch_quiver as torch_qv
from typing import List




def get_csr_from_coo(edge_index):
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    node_count = max(np.max(src), np.max(dst))
    data = np.zeros(dst.shape, dtype=np.int32)
    csr_mat = csr_matrix(
        (data, (edge_index[0].numpy(), edge_index[1].numpy())))
    return csr_mat


class CSRTopo:
    """Graph topology in CSR format.
    
    ```python
    >>> csr_topo = CSRTopo(edge_index=edge_index)
    >>> csr_topo = CSRTopo(indptr=indptr, indices=indices)
    ```
    
    Args:
        edge_index ([torch.LongTensor], optinal): edge_index tensor for graph topo
        indptr (torch.LongTensor, optinal): indptr for CSR format graph topo
        indices (torch.LongTensor, optinal): indices for CSR format graph topo
    """
    def __init__(self, edge_index=None, indptr=None, indices=None, eid=None):
        if edge_index is not None:
            csr_mat = get_csr_from_coo(edge_index)
            self.indptr_ = torch.from_numpy(csr_mat.indptr).type(torch.long)
            self.indices_ = torch.from_numpy(csr_mat.indices).type(torch.long)
        elif indptr is not None and indices is not None:
            if isinstance(indptr, torch.Tensor):
                self.indptr_ = indptr.type(torch.long)
                self.indices_ = indices.type(torch.long)
            elif isinstance(indptr, np.ndarray):
                self.indptr_ = torch.from_numpy(indptr).type(torch.long)
                self.indices_ = torch.from_numpy(indices).type(torch.long)
        self.eid_ = eid
        self.feature_order_ = None

    @property
    def indptr(self):
        """Get indptr

        Returns:
            torch.LongTensor: indptr 
        """
        return self.indptr_

    @property
    def indices(self):
        """Get indices

        Returns:
            torch.LongTensor: indices
        """
        return self.indices_

    @property
    def eid(self):
        return self.eid_

    @property
    def feature_order(self):
        """Get feature order for this graph

        Returns:
            torch.LongTensor: feature order 
        """
        return self.feature_order_

    @feature_order.setter
    def feature_order(self, feature_order):
        """Set feature order

        Args:
            feature_order (torch.LongTensor): set feature order
        """
        self.feature_order_ = feature_order

    @property
    def degree(self):
        """Get degree of each node in this graph

        Returns:
            [torch.LongTensor]: degree tensor for each node
        """
        return self.indptr[1:] - self.indptr[:-1]

    @property
    def node_count(self):
        """Node count of the graph

        Returns:
            int: node count
        """
        return self.indptr_.shape[0] - 1

    @property
    def edge_count(self):
        """Edge count of the graph

        Returns:
            int: edge count
        """
        return self.indices_.shape[0]

    
    def share_memory_(self):
        """
        Place this CSRTopo in shared memory
        """
        self.indptr_.share_memory_()
        self.indices_.share_memory_()
        if self.eid_ is not None:
            self.eid_.share_memory_()
        
        if self.feature_order_ is not None:
            self.feature_order_.share_memory_()



def reindex_by_config(adj_csr: CSRTopo, graph_feature, gpu_portion):

    node_count = adj_csr.indptr.shape[0] - 1
    total_range = torch.arange(node_count, dtype=torch.long)
    perm_range = torch.randperm(int(node_count * gpu_portion))
    # sort and shuffle
    degree = adj_csr.indptr[1:] - adj_csr.indptr[:-1]
    _, prev_order = torch.sort(degree, descending=True)
    new_order = torch.zeros_like(prev_order)
    prev_order[:int(node_count * gpu_portion)] = prev_order[perm_range]
    new_order[prev_order] = total_range
    graph_feature = graph_feature[prev_order]
    return graph_feature, new_order


def reindex_feature(graph: CSRTopo, feature, ratio):
    assert isinstance(graph, CSRTopo), "Input graph should be CSRTopo object"
    feature, new_order = reindex_by_config(graph, feature, ratio)
    return feature, new_order


UNITS = {
    #
    "KB": 2**10,
    "MB": 2**20,
    "GB": 2**30,
    #
    "K": 2**10,
    "M": 2**20,
    "G": 2**30,
}


def parse_size(sz) -> int:
    if isinstance(sz, int):
        return sz
    elif isinstance(sz, float):
        return int(sz)
    elif isinstance(sz, str):
        for suf, u in sorted(UNITS.items()):
            if sz.upper().endswith(suf):
                return int(float(sz[:-len(suf)]) * u)
    raise Exception("invalid size: {}".format(sz))