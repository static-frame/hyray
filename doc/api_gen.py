import numpy as np
import cupy as cp
import inspect

SKIP = frozenset((
    '__class__',
    '__delitem__',
    '__getattribute__',
    '__init__',
    '__init_subclass__',
    '__new__',
    '__reduce__',
    '__reduce_ex__',
    '__setattr__',
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
    '__len__',
    '__argmax__',
    '__argmin__',
))

def func_instance(attr, obj):
    '''inspect.signature works for most of these interfaces.
    '''
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

def func_method(attr, obj):
    '''inspect.signature does not work for most of these interfaces.
    '''
    doc = obj.__doc__
    if doc is None:
        print('# no docstr:', attr)
        return

    sig_str = doc.split('\n\n')[0][2:] # drop leading "a."
    sig_str = sig_str.replace('[', '')
    sig_str = sig_str.replace(']', '')
    args_raw = sig_str[sig_str.find('(')+1: sig_str.find(')')]

    args_names = []
    for arg in args_raw.split(','):
        if not arg:
            continue
        if '=' in arg:
            args_names.append(arg.split('=')[0].strip())
        else:
            args_names.append(arg.strip())

    args_assign = []
    positional = True
    for name in args_names:
        if name in ('*', '/'):
            positional = False
        elif positional:
            args_assign.append(name)
        else:
            args_assign.append(f'{name}={name}')

    call_str = f'{attr}({", ".join(args_assign)})'

    # if we have args, add self with a comma
    if not args_names:
        def_str = sig_str.replace(f'{attr}(', f'{attr}(self')
    else:
        def_str = sig_str.replace(f'{attr}(', f'{attr}(self, ')

    if attr in NO_WRAP:
        print(f'''
    def {def_str}:
        return self._array.{call_str}''')
    else:
        print(f'''
    def {def_str}:
        return CuArray(self._array.{call_str})''')




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

    # print('\n### functions magic')
    # for attr, obj in functions_magic:
    #     func_instance(attr, obj)

    print('\n### functions')
    for attr, obj in functions:
        func_method(attr, obj)




if __name__ == '__main__':
    gen_instance_attrs()