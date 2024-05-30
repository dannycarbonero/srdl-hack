import platform
from pathlib import Path

def get_parent_path(context, subdirectory = None, make = False):
    os = platform.system()
    computer = platform.node()

    if context == 'data':

        if 'Main' in computer:

            if os == 'Windows':
                parent_path = 'E:/Neural Data/'
            else:
                parent_path = '/home/warehaus/Neural Data/'

        elif 'Mobile' in computer:
            parent_path = 'D:/Neural Data/'

        elif 'Doc' in computer:

            if os == 'Windows':
                parent_path = 'D:/Neural Data/'
            else:
                parent_path = '/home/SSD 1/Neural Data/'

        else:
            parent_path = '/projectnb/ecog/dcarbo/'

    elif context == 'scc':
        parent_path = '/projectnb/ecog/dcarbo/'

    else:

        if os == 'Windows':
            parent_path = 'C:/Users/C/C.ode/'
        else:
            parent_path = '/home/c/C.ode/'


    if subdirectory:
        parent_path = parent_path + subdirectory + '/'

    if make:
        Path(parent_path).mkdir(exist_ok = True)

    return parent_path
