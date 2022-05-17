import torch_quiver as torch_qv
import torch
# import shard_node as sn


__all__ = ["ShardNode"]
# 现在需要决定要不要复用 quiver class 中的代码
# 复用的话，需要修改 quiver class 中的代码，可以添加 indptr cpu 部分，也可以完全不分配 indptr 内存
# 不复用的话，将 quiver class 中的代码抄一份再进行修改

# 将 ShardNode 设计为和所有 gpu 内存分配相关的接口
# 也是就是说，如果需要分配 gpu 内存的话，ShardNode 完全可以取代 quiver
# 因此 ShardNode 应该具有如下方法：
#   device_quiver_from_csr_array, sample_neighbor, reindex_single(这个函数似乎并不需要重写)
class ShardNode:
    def __init__(self, current_device = 0, memory_budget = 0):
        self.current_device = current_device
        # self.shard_node = torch_qv.ShardNode(current_device)
        # 这里的 memory_budget 不需要再调用 parse_size，假设上层已经调用了
        self.memory_budget = memory_budget
        self.cpu_tensor = None
        self.shard_node = None
    
    def append(self, node, device):
        if self.shard_node is None:
            raise Exception(f"shard_node does not construct")
        
        self.shard_node.append(node, device)
        # 下面都是考虑了多 gpu 情况的，此处暂时不予考虑
        # 需要添加对应 cpu cuda 底层分配数据的接口
        # if device == -1:
            # 添加到 cpu 中
            # if self.cpu_tensor is not None:
            #     raise Exception("cpu tensor has been already appended")
            # self.cpu_tensor = node
            # self.shard_node.append(node, -1)
        # 这个 if 没有用，此处不考虑多 gpu 情况
        # if self.shard_node_config.device_memory_budget.get(device, None) is None:
            # 添加到 gpu 中
        # if device >= 0:
        #     self.shard_node.append(node, device)
        # else:
        #     raise Exception(f"{device} node has been already appended")
        
    def partition(self, indices: torch.Tensor, memory_budget: int):
        # 计算出有多少内存在 cpu gpu 中
        cache_size = memory_budget // 8
        return [indices[:cache_size], indices[cache_size:]]

    def device_quiver_from_csr_array(self, indptr, indices, edge_idx, device, memory_budget):
        # 分配 indptr, edge_idx 内存
        self.shard_node = torch_qv.device_node_from_csr_array(indptr, indices, edge_idx, memory_budget, device)
        # 分配 indices 内存
        gpu_part, self.cpu_part = self.partition(indices, memory_budget)
        if gpu_part.shape[0] > 0:
            self.append(gpu_part, device)
        if self.cpu_part.numel() > 0:
            self.append(self.cpu_part, -1)
        
            

    def sample_neighbor(self, stream, node, size):
        # 进行实际的采样
        # 采样的时候会使用底层提供的接口进行采样
        n_id, count = self.shard_node.sample_neighbor(stream, node, size)
        return n_id, count
    
    # def reindex(self, inputs, outputs, counts):
    #     return self.shard_node.reindex_single(inputs, outputs, counts)
    