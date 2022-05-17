import torch_quiver as torch_qv
import torch

from .utils import parse_size


class Offset:
    def __init__(self, start, end):
        self.start_ = start
        self.end_ = end

    @property
    def start(self):
        return self.start_

    @property
    def end(self):
        return self.end_


class ShardTensorConfig:
    """
    """
    def __init__(self, device_memory_budget):
        self.tensor_offset_device = {}

        self.device_memory_budget = device_memory_budget
        for device in device_memory_budget:
            self.device_memory_budget[device] = parse_size(self.device_memory_budget[device])

    @property
    def device_list(self):

        return list(self.device_memory_budget.keys())


class ShardTensor:
    """[summary]
    """
    def __init__(self, current_device: int,
                 shard_tensor_config: ShardTensorConfig):
        self.shard_tensor = torch_qv.ShardTensor(current_device)
        self.current_device = current_device
        self.shard_tensor_config = shard_tensor_config or ShardTensorConfig({})
        self.topo = None
        self.cpu_tensor = None


    def append(self, cpu_tensor, device):
        if device == -1:
            if self.cpu_tensor is not None:
                raise Exception("cpu tensor has been already appended")
            self.cpu_tensor = cpu_tensor
            self.shard_tensor.append(cpu_tensor, -1)
            return
        if self.shard_tensor_config.device_memory_budget.get(device,
                                                             None) is None:
            self.shard_tensor_config.tensor_offset_device[device] = Offset(
                self.shard_tensor.size(0),
                self.shard_tensor.size(0) + cpu_tensor.shape[0])
            self.shard_tensor_config.device_memory_budget[
                device] = cpu_tensor.numel() * 4
            print(
                f"LOG >>> Memory Budge On {device} is {self.shard_tensor_config.device_memory_budget[device] // 1024 // 1024} MB"
            )
            self.shard_tensor.append(cpu_tensor, device)
        else:
            raise Exception(f"{device} tensor has been already appended")

    def partition(self, tensor, memory_budget):
        """
        Args:
            tensor: pytorch cpu tensor
            memory_budget: memory size in bytes
            
        """
        # 暂时先假设为float tensor
        element_size = tensor.shape[1] * 4
        return memory_budget // element_size

    def from_cpu_tensor(self, tensor):
        cur_pos = 0
        size = 0
        for device_id, memory_budget in self.shard_tensor_config.device_memory_budget.items(
        ):
            if cur_pos > tensor.shape[0]:
                break

            size = self.partition(tensor, memory_budget)
            size = min(size, tensor.shape[0] - cur_pos)
            self.shard_tensor.append(tensor[cur_pos:cur_pos + size], device_id)
            device_offset = Offset(cur_pos, cur_pos + size)
            self.shard_tensor_config.tensor_offset_device[
                device_id] = device_offset

            cur_pos += size
            print(
                f"LOG >>> Assign {int(100 * size * 1.0 / tensor.shape[0])}% data to {device_id}"
            )

        if cur_pos < tensor.shape[0]:
            # allocate the rest of data on CPU
            self.cpu_tensor = tensor[cur_pos:]
            self.shard_tensor.append(self.cpu_tensor, -1)
            print(
                f"LOG >>> Assign {100 - int(100 * cur_pos * 1.0 / tensor.shape[0])}% data to CPU"
            )
            del tensor


    def __getitem__(self, nodes):
        nodes = nodes.to(self.current_device)
        feature = self.shard_tensor[nodes]

        return feature

    @property
    def shape(self):
        return self.shard_tensor.shape()

    @property
    def device(self):
        return self.current_device

    def size(self, dim):
        return self.shard_tensor.size(dim)
