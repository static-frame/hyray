import numpy as np
import cupy as cp
import inspect

SKIP = frozenset((
    '__class__',
    '__delitem__',
    '__init__',
    '__init_subclass__',
    '__new__',
    '__reduce__',
    '__reduce_ex__',
    '__setattr__',
    '__setitem__',
    '__subclasshook__',
))

NO_WRAP = frozenset((
    '__bool__',
    '__complex__',
    '__delattr__',
    '__dir__',
    '__sizeof__',
    '__str__',
    '__dlpack__',
    '__dlpack_device__',
    '__float__',
    '__format__',
    '__getattribute__',
    '__int__',
    '__repr__',
    '__str__',
))

def func_instance(attr, obj):
    sig_str = '(self)'
    try:
        sig = inspect.signature(obj)
        args = len(sig.parameters)
        # call_args = ', '.join(f'{k}={k}' for k in sig.parameters.keys())
        call_args = ', '.join(sig.parameters.keys())
        if args > 0:
            sig_str = str(sig).replace('(', '(self, ')
    except ValueError: # signature might
        args = 0
        call_args = ''
    if attr in NO_WRAP:
        print(f'''
    def {attr}{sig_str}:
        return self._array.{attr}({call_args})''')
    else:
        print(f'''
    def {attr}{sig_str}:
        return CuArray(self._array.{attr}({call_args}))''')


def gen_instance_attrs():
    np_array = np.array(())
    cp_array = cp.array(())

    properties = []
    functions = []
    functions_magic = []

    for attr in sorted(dir(np_array), key=lambda n: n.lower()):
        if not hasattr(cp_array, attr) or attr in SKIP:
            # print(f'not in cupy: {attr}')
            continue

        obj = getattr(np_array, attr)
        if callable(obj):
            if attr.startswith('__'):
                functions_magic.append((attr, obj))
            else:
                functions.append((attr, obj))
        else:
            properties.append((attr, obj))

    # print('\n### properties')
    # for attr in properties:
    #     print(attr)

    print('\n### functions magic')
    for attr, obj in functions_magic:
        func_instance(attr, obj)

    # print('\n### functions')
    # for attr in functions:
    #     print(attr)




if __name__ == '__main__':
    gen_instance_attrs()