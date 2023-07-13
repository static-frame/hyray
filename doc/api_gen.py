import numpy as np
import cupy as cp


def write_property(attr, obj):
    f'''
@property
def {name}(self) -> :
'''

def gen_instance_attrs():
    np_array = np.array(())
    cp_array = cp.array(())

    properties = []
    functions = []

    for attr in sorted(dir(np_array), key=lambda n: n.lower()):
        if not hasattr(cp_array, attr):
            print(f'not in cupy: {attr}')
            continue

        obj = getattr(np_array, attr)
        if callable(obj):
            functions.append(attr)
        else:
            properties.append(attr)

    print('\n### properties')
    for attr in properties:
        print(attr)

    print('\n### functions')
    for attr in functions:
        print(attr)




if __name__ == '__main__':
    gen_instance_attrs()