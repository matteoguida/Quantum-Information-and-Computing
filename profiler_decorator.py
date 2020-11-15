from cProfile import Profile
import pstats

def profile(sort_args=['cumulative'], print_args=[10]):
    '''
    Simple wrapper of cProfile python library to print in stdout useful infos about specific function runtime

    Suggested usage:
    Insert the following line before the definitio of the function whose profiling is needed
        @profile(sort_args=['name'], print_args=[N])
    with N = # of tasks which are listed
    '''
    profiler = Profile()

    def decorator(fn):
        def inner(*args, **kwargs):
            result = None
            try:
                result = profiler.runcall(fn, *args, **kwargs)
            finally:
                stats = pstats.Stats(profiler)
                stats.strip_dirs().sort_stats(*sort_args).print_stats(*print_args)
            return result
        return inner
    return decorator