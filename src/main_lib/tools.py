

def flatten_list(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result



def are_lists_equal(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return set1 == set2