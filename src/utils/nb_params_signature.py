import numpy as np
import numba as nb
from numba import types
from numba.core.types import Optional, unicode_type, Tuple, DictType, ListType
from numba.typed import Dict, List

from src.utils.constants import numba_config

nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]

params_type = DictType(unicode_type, nb_float)
params_list_type = ListType(params_type)
params_dict_type = DictType(unicode_type, nb_float[:])


get_indicator_params_signature = params_type()

get_backtest_params_signature = params_type()

create_params_list_template_signature = Tuple(
    (
        params_list_type,
        params_list_type,
    )
)(nb_int)

create_params_dict_template_signature = Tuple(
    (
        params_dict_type,
        params_dict_type,
    )
)(nb_int)

# -------------------- get_params_list_value 签名 --------------------
# 参数：key (unicode_type), params_list (List[Dict])
# 返回：一个 np.array (nb_float[:])
get_params_list_value_signature = nb_float[:](
    unicode_type,
    params_list_type,
)

# -------------------- set_params_list_value 签名 --------------------
# 参数：key (unicode_type), params_list (List[Dict]), arr (nb_float[:])
set_params_list_value_signature = types.void(
    unicode_type,
    params_list_type,
    nb_float[:],
)

# -------------------- get_params_dict_value 签名 --------------------
# 参数：key (unicode_type), params_dict (Dict[unicode, array])
# 返回：一个 np.array (nb_float[:])
get_params_dict_value_signature = nb_float[:](unicode_type, params_dict_type)

# -------------------- set_params_dict_value 签名 --------------------
# 参数：key (unicode_type), params_dict (Dict[unicode, array]), arr (nb_float[:])
set_params_dict_value_signature = types.void(
    unicode_type, params_dict_type, nb_float[:]
)

# -------------------- convert_params_dict_list 签名 --------------------
# 参数：params_dict (Dict[unicode, array])
# 返回：一个 List[Dict[unicode, float]]
convert_params_dict_list_signature = params_list_type(params_dict_type)

# -------------------- get_data_mapping 签名 --------------------
# 参数：tohlcv_np (nb_float[:, :]), tohlcv_np_mtf (nb_float[:, :])
# 返回：一个包含 int64 数组的 Dict
get_data_mapping_signature = DictType(unicode_type, nb_int[:])(
    Optional(nb_float[:, :]), Optional(nb_float[:, :])
)
