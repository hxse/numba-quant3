import numpy as np
import numba as nb
from numba import types
from numba.core.types import Optional, unicode_type, Tuple, DictType, ListType
from numba.typed import Dict, List

from src.utils.constants import numba_config


nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


params_type = DictType(unicode_type, nb_float)
params_list_type = ListType(params_type)
params_dict_type = DictType(unicode_type, nb_float[:])


get_indicator_params_signature = params_type(nb_bool)


get_backtest_params_signature = params_type(nb_bool)


create_params_list_template_signature = Tuple(
    (
        params_list_type,
        params_list_type,
    )
)(nb_int, nb_bool)


create_params_dict_template_signature = Tuple(
    (
        params_dict_type,
        params_dict_type,
    )
)(nb_int, nb_bool)


get_params_list_value_signature = nb_float[:](
    unicode_type,
    params_list_type,
)


set_params_list_value_signature = types.void(
    unicode_type,
    params_list_type,
    nb_float[:],
)


get_params_dict_value_signature = nb_float[:](unicode_type, params_dict_type)


set_params_dict_value_signature = types.void(
    unicode_type, params_dict_type, nb_float[:]
)


convert_params_dict_list_signature = params_list_type(params_dict_type)
convert_params_list_dict_signature = params_dict_type(params_list_type)


get_data_mapping_signature = DictType(unicode_type, nb_int[:])(
    Optional(nb_float[:, :]), Optional(nb_float[:, :])
)


get_init_tohlcv_signature = DictType(unicode_type, nb_float[:])(
    Optional(nb_float[:, :])
)
get_init_tohlcv_smoothed_signature = DictType(unicode_type, nb_float[:])(
    Optional(nb_float[:, :]),
    Optional(unicode_type),
)
