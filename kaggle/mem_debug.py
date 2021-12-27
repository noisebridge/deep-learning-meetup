from torch.cuda import memory_allocated, max_memory_allocated, memory_reserved, max_memory_reserved, memory_cached

def mem_readout(num=0):
    """
    just printingo out memory information
    """
    
    readout = "memory_allocated " + str(memory_allocated()) + " "
    readout += "max_memory_allocated " + str(max_memory_allocated()) + "\n"
    readout += "memory_reserved " + str(memory_reserved()) + " "
    readout += "max_memory_reserved " + str(max_memory_reserved()) + "\n"
    readout += "memory_cached " + str(memory_cached())
    if num != 0:
        readout = "step " + str(num) + "\n" + readout
    print(readout)
