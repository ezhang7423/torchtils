
SEPARATOR = "\n\n-----------------\n\n"

class colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def red(*args, **kwargs):
    args = list(args)
    args[0] = colors.FAIL + str(args[0])
    print(*args, colors.ENDC, **kwargs)


def green(*args, **kwargs):
    args = list(args)
    args[0] = colors.OKGREEN + str(args[0])
    print(*args, colors.ENDC, **kwargs)


def bold(*args, **kwargs):
    args = list(args)
    args[0] = colors.BOLD + str(args[0])
    print(*args, colors.ENDC, **kwargs)
