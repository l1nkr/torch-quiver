import torch
from quiver.shard_tensor import ShardTensor, ShardTensorConfig
from quiver.utils import reindex_feature, CSRTopo, parse_size
from typing import List
import numpy as np
from torch._C import device

__all__ = ["Feature", "DeviceConfig"]


class DeviceConfig:
    def __init__(self, gpu_parts, cpu_part):
        self.gpu_parts = gpu_parts
        self.cpu_part = cpu_part


class Feature(object):
    """Feature partitions data onto different GPUs' memory and CPU memory and does feature collection with high performance.
    You will need to set `device_cache_size` to tell Feature how much data it can cached on GPUs memory. By default, it will partition data by your  `device_cache_size`, if you want to cache hot data, you can pass
    graph topology `csr_topo` so that Feature will reorder all data by nodes' degree which we expect to provide higher cache hit rate and will offer better performance with regard to cache random data.
    
    ```python
    >>> cpu_tensor = torch.load("cpu_tensor.pt")
    >>> feature = Feature(0, device_list=[0, 1], device_cache_size='200M')
    >>> feature.from_cpu_tensor(cpu_tensor)
    >>> choose_idx = torch.randint(0, feature.size(0), 100)
    >>> selected_feature = feature[choose_idx]
    ```
    Args:
        rank (int): device for feature collection kernel to launch
        device_list ([int]): device list for data placement
        device_cache_size (Union[int, str]): cache data size for each device, can be like `0.9M` or `3GB`
        cache_policy (str, optional): cache_policy for hot data, can be `device_replicate` or `p2p_clique_replicate`, choose `p2p_clique_replicate` when you have NVLinks between GPUs, else choose `device_replicate`. (default: `device_replicate`)
        csr_topo (quiver.CSRTopo): CSRTopo of the graph for feature reordering
        
    """
    def __init__(self,
                 rank: int,
                 device_list: List[int],
                 device_cache_size: int = 0,
                 csr_topo: CSRTopo = None):
        self.device_cache_size = device_cache_size
        self.cache_policy = 'device_replicate'
        self.device_list = device_list
        self.device_tensor_list = {}
        self.clique_tensor_list = {}
        self.rank = rank
        # self.topo = Topo(self.device_list)
        self.csr_topo = csr_topo
        self.feature_order = None

  
    def cal_size(self, cpu_tensor: torch.Tensor, cache_memory_budget: int):
        element_size = cpu_tensor.shape[1] * 4
        cache_size = cache_memory_budget // element_size
        return cache_size

    def partition(self, cpu_tensor: torch.Tensor, cache_memory_budget: int):

        cache_size = self.cal_size(cpu_tensor, cache_memory_budget)
        return [cpu_tensor[:cache_size], cpu_tensor[cache_size:]]


    def from_cpu_tensor(self, cpu_tensor: torch.Tensor):
        """Create quiver.Feature from a pytorh cpu float tensor

        Args:
            cpu_tensor (torch.FloatTensor): input cpu tensor
        """
        if self.cache_policy == "device_replicate":
            cache_memory_budget = parse_size(self.device_cache_size)
            shuffle_ratio = 0.0

        print(
            f"LOG>>> {min(100, int(100 * cache_memory_budget / cpu_tensor.numel() / 4))}% data cached"
        )
        if self.csr_topo is not None:
            if self.csr_topo.feature_order is None:
                cpu_tensor, self.csr_topo.feature_order = reindex_feature(
                    self.csr_topo, cpu_tensor, shuffle_ratio)
            self.feature_order = self.csr_topo.feature_order.to(self.rank)
        cache_part, self.cpu_part = self.partition(cpu_tensor,
                                                   cache_memory_budget)
        self.cpu_part = self.cpu_part.clone()
        if cache_part.shape[0] > 0 and self.cache_policy == "device_replicate":
            for device in self.device_list:
                shard_tensor = ShardTensor(self.rank, ShardTensorConfig({}))
                shard_tensor.append(cache_part, device)
                self.device_tensor_list[device] = shard_tensor
        # 构建CPU Tensor
        if self.cpu_part.numel() > 0:
            if self.cache_policy == "device_replicate":
                shard_tensor = self.device_tensor_list.get(
                    self.rank, None) or ShardTensor(self.rank,
                                                    ShardTensorConfig({}))
                shard_tensor.append(self.cpu_part, -1)
                self.device_tensor_list[self.rank] = shard_tensor


    def set_local_order(self, local_order):
        """ Set local order array for quiver.Feature

        Args:
            local_order (torch.Tensor): Tensor which contains the original indices of the features

        """
        local_range = torch.arange(end=local_order.size(0),
                                   dtype=torch.int64,
                                   device=self.rank)
        self.feature_order = torch.zeros_like(local_range)
        self.feature_order[local_order.to(self.rank)] = local_range

    def __getitem__(self, node_idx: torch.Tensor):
        node_idx = node_idx.to(self.rank)
        if self.feature_order is not None:
            node_idx = self.feature_order[node_idx]
        if self.cache_policy == "device_replicate":
            shard_tensor = self.device_tensor_list[self.rank]
            return shard_tensor[node_idx]

    def size(self, dim: int):
        """ Get dim size for quiver.Feature

        Args:
            dim (int): dimension 

        Returns:
            int: dimension size for dim
        """
        if self.cache_policy == "device_replicate":
            shard_tensor = self.device_tensor_list[self.rank]
            return shard_tensor.size(dim)


    def dim(self):
        """ Get the number of dimensions for quiver.Feature

        Args:
            None

        Returns:
            int: number of dimensions
        """
        return len(self.shape)

    @property
    def shape(self):
        if self.cache_policy == "device_replicate":
            shard_tensor = self.device_tensor_list[self.rank]
            return shard_tensor.shape


