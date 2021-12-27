from torch.cuda import memory_allocated, max_memory_allocated, memory_reserved, max_memory_reserved, memory_cached

def convert(num, scale="G"):
    """
    Just converting memory #s to something readable
    """
    if scale == "M":
        num = num/(1024**2)
        formatted_num =  "{:.2f}".format(num)
        return formatted_num + "MB"
    if scale == "G":
        num = num/(1024**3)
        formatted_num =  "{:.2f}".format(num)
        return formatted_num + "GB"

def mem_readout(num=0):
    """
    just printingo out memory information
    """
    
    readout = "memory_allocated " + convert(memory_allocated()) + " "
    readout += "max_memory_allocated " + convert(max_memory_allocated()) + "\n"
    readout += "memory_reserved " + convert(memory_reserved()) + " "
    readout += "max_memory_reserved " + convert(max_memory_reserved()) + "\n"
    readout += "memory_cached " + convert(memory_cached())
    if num != 0:
        readout = "step " + str(num) + "\n" + readout
    print(readout)
