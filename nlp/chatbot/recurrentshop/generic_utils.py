import types
import marshal


# Function serialization / deserialixation


def func_dump(func):
    '''Serialize user defined function.'''

    code = marshal.dumps(func.__code__).decode('raw_unicode_escape')
    defaults = func.__defaults__
    if func.__closure__:
        closure = tuple(c.cell_contents for c in func.__closure__)
    else:
        closure = None
    return (code, defaults, closure)


def func_load(
        code,
        defaults=None,
        closure=None,
        globs=None,
):
    '''Deserialize user defined function.'''

    if isinstance(code, (tuple, list)):  # unpack previous dump
        (code, defaults, closure) = code
    code = marshal.loads(code.encode('raw_unicode_escape'))
    if closure is not None:
        closure = func_reconstruct_closure(closure)
    if globs is None:
        globs = globals()
    return types.FunctionType(code, globs, name=code.co_name,
                              argdefs=defaults, closure=closure)


def func_reconstruct_closure(values):
    '''Deserialization helper that reconstructs a closure.'''

    nums = range(len(values))
    src = ['def func(arg):']
    src += ['  _%d = arg[%d]' % (n, n) for n in nums]
    src += ['  return lambda:(%s)' % ','.join(['_%d' % n for n in
                                               nums]), '']
    src = '\n'.join(src)
    try:
        exec(src, globals())
    except:
        raise SyntaxError(src)
    return func(values).__closure__


def serialize_function(func):
    if isinstance(func, types.LambdaType):
        function = func_dump(func)
        function_type = 'lambda'
    else:
        function = func.__name__
        function_type = 'function'
    return (function_type, function)


def deserialize_function(txt):
    (function_type, function) = txt
    if function_type == 'function':
        return globals()[function]
    else:
        return func_load(function, globs=globals())
