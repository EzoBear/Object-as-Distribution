from termcolor import cprint

def print_info(info, _type=None):
    if _type is not None:
        if isinstance(info, str):
            cprint(info, _type[0], attrs=[_type[1]])
        elif isinstance(info, list):
            for i in range(info):
                cprint(i, _type[0], attrs=[_type[1]])
    else:
        print(info)