import torch
def translate_node(mask):
    """
    translates node position to mask position
    """
    build_map = {}
    j = 0
    for i,m in enumerate(mask):
        if m:
            build_map[i] = j
            j+=1

    return build_map

def translate_mask(mask):
    """
    translates mask position to node position
    """
    build_map = []
    for i,m in enumerate(mask):
        if m:
            build_map.append(i)
    return(build_map)

    return build_map
def subset_mask_map(mask, sub_mask):
    """
    im too lazy to explain //TODO
    """
    tmap = translate_node(mask)
    map_mask = []
    for i,m in enumerate(sub_mask):
        if m:
            map_mask.append(tmap[i])

    return map_mask

def l2(x,y):
    return torch.sqrt(torch.sum(torch.pow(torch.sub(x,y),2),dim=0))

def rbf(x,y,sigma=100):
    return torch.exp(-0.5 * torch.sum(torch.pow(torch.sub(x,y), 2)) / sigma ** 2)

if __name__ == '__main__':
    print(subset_mask_map([True, True, False, False, True, True, True], [False, True, False, False, True, False, True])) # expect 1,2,4
