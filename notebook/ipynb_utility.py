import ipynbname
    
import random
import numpy as np
import torch as torch

def get_file():
    '''
    尝试获取 ipynb 文件路径
    需要安装模块 
    * ipynbname (一般情况)
    * IPython (vscode)
    '''

    # 一般情况
    try:
        return ipynbname.path()
    except:
        pass

    # vscode
    try:
        # ref: https://github.com/msm1089/ipynbname/issues/17#issuecomment-1293269863
        from IPython.core.getipython import get_ipython
        ip = get_ipython()
        if ip and '__vsc_ipynb_file__' in ip.user_ns:
            return ip.user_ns['__vsc_ipynb_file__']
    except:
        pass

    raise Exception("Can not find the path of .ipynb")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)