import numpy as np
import numba as nb
from numba import types
from numba.core.types import Optional, unicode_type, Tuple, DictType, ListType
from numba.typed import Dict, List

from src.utils.constants import numba_config


nb_int = numba_config["nb"]["int"]
nb_float = numba_config["nb"]["float"]
nb_bool = numba_config["nb"]["bool"]


dict_float_type = DictType(unicode_type, nb_float)
list_dict_float_type = ListType(dict_float_type)
list_list_dict_float_type = ListType(ListType(dict_float_type))
dict_float_1d_type = DictType(unicode_type, nb_float[:])
dict_int_1d_type = DictType(unicode_type, nb_int[:])
list_dict_float_1d_type = ListType(dict_float_1d_type)


get_indicator_params_signature = dict_float_type(nb_bool)
get_backtest_params_signature = dict_float_type(nb_bool)


create_indicator_params_list_signature = list_list_dict_float_type(
    nb_int, nb_int, nb_bool
)
create_backtest_params_list_signature = list_dict_float_type(nb_int, nb_bool)


create_params_dict_template_signature = Tuple(
    (
        dict_float_1d_type,
        dict_float_1d_type,
    )
)(nb_int, nb_bool)


get_params_list_value_signature = nb_float[:](
    unicode_type,
    list_dict_float_type,
)


set_params_list_value_signature = types.void(
    unicode_type,
    list_dict_float_type,
    nb_float[:],
)

set_params_list_value_mtf_signature = types.void(
    nb_int,
    unicode_type,
    list_list_dict_float_type,
    nb_float[:],
)


get_params_dict_value_signature = nb_float[:](unicode_type, dict_float_1d_type)


set_params_dict_value_signature = types.void(
    unicode_type, dict_float_1d_type, nb_float[:]
)


convert_params_dict_list_signature = list_dict_float_type(dict_float_1d_type)
convert_params_list_dict_signature = dict_float_1d_type(list_dict_float_type)


get_data_mapping_signature = dict_int_1d_type(
    Optional(nb_float[:, :]), Optional(nb_float[:, :])
)

get_data_mapping_mtf_signature = dict_int_1d_type(list_dict_float_1d_type)


get_init_tohlcv_signature = DictType(unicode_type, nb_float[:])(
    Optional(nb_float[:, :])
)
get_init_tohlcv_smoothed_signature = DictType(unicode_type, nb_float[:])(
    nb_float[:, :],
    unicode_type,
)
