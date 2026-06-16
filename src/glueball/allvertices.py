import numpy as np
from itertools import permutations

'''
THIS MODULE GENERATES ALL POSSIBLE CONFIGURATIONS OF N GLUON TO M GLUON PROCESS IN COLOR SPACE. 
TESTED WITH THE TWO GLUON TO THREE GLUON PROCESS. THE RESULT MATCHES PREVIOUS CODE. 

FUNCTIONS:
partition(elements): fully divide incoming / outgoing elements into subsets. elements are an ID list of all incoming/outgoing gluons. 
get_vertex_type(inc_size, out_size): 3-, 4-gluon vertex or spectator. inc_size = # of particles coming into one vertex. same with out_size. 
pair_partitions(inc_partition, out_partition): paired incoming partition results (which is generated using parition(elements) function) with outgoing ones. 
generate_vertex_plans(inc_n, out_n): the most important one. generating ID list according to input num of inc/out gluons, entering ID list to partition, doing pairing and check vtype, return a list of dicts.
format/save: readability/document.  
'''

# partition function divides a set of n elements into subsets and lists all possible ways.
def partition(elements): # CHECKED
    n = len(elements)
    result = []

    def go(index, current):
        if index == n:
            result.append([subset[:] for subset in current])
            return

        elem = elements[index]

        # choice 1: go to a new subset
        current.append([elem])
        go(index + 1, current)
        current.pop()

        # choice 2: go to an existing subset
        for i in range(len(current)):
            current[i].append(elem)
            go(index + 1, current)
            current[i].pop()

    go(0, [])
    return result



def get_vertex_type(inc_size, out_size): # CHECKED
    if inc_size == 1 and out_size == 1:
        return "spectator"
    elif (inc_size == 2 and out_size == 1) or (inc_size == 1 and out_size == 2):
        return "3-gluon vertex"
    elif inc_size == 2 and out_size == 2:
        return "4-gluon vertex"
    else:
        return "invalid"


def pair_partitions(inc_partition, out_partition): #CHECKED

    # Number of subsets must match; otherwise some gluons are left unpaired
    if len(inc_partition) != len(out_partition):
        return []

    valid_pairings = []
    n = len(inc_partition)

    # Try every permutation of outgoing subsets to match with incoming subsets
    for perm in permutations(range(n)): # which gets all permutations of perm = P[0, 1, ..., n - 1] = (1, 0, ...) or (2, 0, 1, ...)...all possible permutations, where n is the # of inc_partition. for example, if we have a specific permutation perm[0] = 1 it means that we pair the 0-th incoming with the 1st outgoing.   
        pairing = []
        is_valid = True
        
        for i in range(n):
            inc_subset = inc_partition[i]
            out_subset = out_partition[perm[i]]
            vtype = get_vertex_type(len(inc_subset), len(out_subset))
            
            if vtype == "invalid":
                is_valid = False
                break
            
            pairing.append({
                'incoming': inc_subset,
                'outgoing': out_subset,
                'type': vtype
            })
        
        if is_valid:
            valid_pairings.append(pairing)
    
    return valid_pairings

# IMPORTANT===============================================
def generate_vertex_plans(inc_n, out_n): # parameters entered in main. # 'in' first
    """    
    Returns:
        list of dicts, each representing a valid configuration:
        {'id': int, 'vertices': [{'incoming': [...], 'outgoing': [...], 'type': '...'}, ...]}
    """
    # Generate all partitions for both sides
    inc_elements = list(range(1, inc_n + 1)) # generate the no. for elements: [1, 2, 3, ..., inc_n +1]
    out_elements = list(range(1, out_n + 1))
    #print(inc_elements)
    
    inc_partitions = partition(inc_elements)
    out_partitions = partition(out_elements)
    
    all_plans = [] # output
    plan_id = 1 # assign an ID for each plan
    
    # Pair every incoming partition with every outgoing partition
    for inc_p in inc_partitions:
        for out_p in out_partitions:
            pairings = pair_partitions(inc_p, out_p)
            for pairing in pairings:
                plan = {
                    'id': plan_id,
                    'vertices': pairing
                } 
                all_plans.append(plan)
                plan_id += 1
    
    return all_plans
# ===================================

def format_plan(plan): # CHECKED # this parameter is A plan but not ALL plans 
    lines = [f"config {plan['id']}:\n"]
    for v_idx, v in enumerate(plan['vertices'], start=1): # v_idx = no. of vertex; v = dict of valid incoming-outgoing pair
        lines.append(f"    vertex {v_idx}: Incoming {v['incoming']} + Outgoing {v['outgoing']} ; vtype = {v['type']}\n")
    return "".join(lines) #stick strs in list 'lines' together, connecting with space "".  

'''
def save_vertex_plans(plans, filename='vertex_plans.txt'): # UNCHECKED
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Total valid vertex configurations: {len(plans)}\n")
        f.write("=" * 60 + "\n\n")
        
        for plan in plans:
            f.write(format_plan(plan))
            f.write("\n")
    
    print(f"Saved {len(plans)} configurations to '{filename}'")
'''


if __name__ == "__main__":
    test = generate_vertex_plans(2, 3)
    #save_vertex_plans(test, 'test_vertex_plans.txt')
    print("\n--- Sample output for 2 incoming + 3 outgoing ---")
    for i in range(len(test)):
        print(format_plan(test[i]))
    '''
    --- Sample output for 2 incoming + 3 outgoing ---
config 1:
    vertex 1: Incoming [1] + Outgoing [1, 3] ; vtype = 3-gluon vertex
    vertex 2: Incoming [2] + Outgoing [2] ; vtype = spectator

config 2:
    vertex 1: Incoming [1] + Outgoing [2] ; vtype = spectator
    vertex 2: Incoming [2] + Outgoing [1, 3] ; vtype = 3-gluon vertex

config 3:
    vertex 1: Incoming [1] + Outgoing [1] ; vtype = spectator
    vertex 2: Incoming [2] + Outgoing [2, 3] ; vtype = 3-gluon vertex

config 4:
    vertex 1: Incoming [1] + Outgoing [2, 3] ; vtype = 3-gluon vertex
    vertex 2: Incoming [2] + Outgoing [1] ; vtype = spectator

config 5:
    vertex 1: Incoming [1] + Outgoing [1, 2] ; vtype = 3-gluon vertex
    vertex 2: Incoming [2] + Outgoing [3] ; vtype = spectator

config 6:
    vertex 1: Incoming [1] + Outgoing [3] ; vtype = spectator
    vertex 2: Incoming [2] + Outgoing [1, 2] ; vtype = 3-gluon vertex
    '''
