from operator import index

import pandas as pd
from collections.abc import MutableMapping
from texttable import Texttable
import latextable

"""
These scripts are intended to make better tables for latex from data
"""

def tupled_flatten_dict(d_dict, parent_key=None):
    items = []
    for key, value in d_dict.items():
        new_key = tuple(parent_key + [key]) if parent_key else tuple([key])
        if isinstance(value, MutableMapping):
            items.extend(tupled_flatten_dict(value, list(new_key)).items())
        else:
            items.append((new_key, value))
    return dict(items)


def dict_to_table(data_dict):
    """
    The function converts data dictionary to a latex table with heirarchical headings and subheadings as necessary.
    the first level of keys define the rows. The second level onwards define the column heading heirarchy.

    :param data_dict:
    :return:
    """
    for key in data_dict.keys():
        if isinstance(data_dict[key], dict):
            data_dict[key] = tupled_flatten_dict(data_dict[key])

    rows = []
    col_keys = None
    for key in data_dict.keys():
        local_row = [key]
        if col_keys is None:
            col_keys = data_dict[key].keys()
            rows.append([ "Model" ] + [ k[0] for k in col_keys ])
        for col_key in col_keys:
            cell_val = data_dict[key][col_key]
            if not isinstance(cell_val, str):
                print(cell_val, format(cell_val, '.2f'))
                cell_val = format(cell_val, '.2f')
            local_row.append(cell_val)
        rows.append(local_row)
    # print(rows)
    table = Texttable()
    table.set_precision(2)
    table.set_deco(Texttable.HEADER)
    table.set_cols_align(["l"] + ["c"]*len(col_keys))
    table.set_cols_dtype(["t"]*(len(col_keys)+1))
    table.add_rows(rows)
    print(table.draw())
    return latextable.draw_latex(table, use_booktabs=True)


if __name__ == "__main__":
    # ## LRA
    # data_dict = {
    #     "S5 (-)": {
    #         "ListOps": 62.15,
    #         "Text": 89.31,
    #         "Retrieval": 91.4,
    #         "Image": 88.0,
    #         "Pathfinder": 95.33,
    #         "avg": 87.46
    #     },
    #     "S4 (-)": {
    #         "ListOps": 59.6,
    #         "Text": 86.82,
    #         "Retrieval": 90.90,
    #         "Image": 88.65,
    #         "Pathfinder": 90.90,
    #         "avg": 86.09
    #     },
    #     "Transformer (-)": {
    #         "ListOps": 36.37,
    #         "Text": 64.27,
    #         "Retrieval": 57.46,
    #         "Image": 42.44,
    #         "Pathfinder": 71.4,
    #         "avg": 54.39
    #     },
    #     "mamba (-)": {
    #         "ListOps": "38.02 (57.3 me)",
    #         "Text": 81.34,
    #         "Retrieval": 80.50,
    #         "Image": 65.08,
    #         "Pathfinder": 69.26,
    #         "avg": 66.84
    #     },
    #     "SSM-au (45.8K)": {
    #         "ListOps": "58.9 (val)",
    #         "Text": "-",
    #         "Retrieval": "-",
    #         "Image": "-",
    #         "Pathfinder": "-",
    #         "avg": "-"
    #     },
    #     "SSM-au (1.5K)": {
    #         "ListOps": "-",
    #         "Text": "-",
    #         "Retrieval": "-",
    #         "Image": "-",
    #         "Pathfinder": "-",
    #         "avg": "-"
    #     }
    # }

    # # MLRegtest different dataset sizes
    # data_dict = {
    #     "RNN (17.1K)": {
    #         "Small": 73.6, "Mid": 79.6, "Large": 83.9
    #     },
    #     "GRU (13.4K)": {
    #         "Small": 84.3, "Mid": 93.4, "Large": 93.9
    #     },
    #     "LSTM (550K)": {
    #         "Small": 72.0, "Mid": 85.5, "Large": 90.1
    #     },
    #     "Transformer (1095K)": {
    #         "Small": 77.9, "Mid": 83.0, "Large": 86.7
    #     },
    #     "Mamba (66.6K)": {
    #         "Small": 80.68, "Mid": 97.5, "Large": 99.94
    #     },
    #     "SSM-au (49.8K)": {
    #         "Small": 83.32, "Mid": 99.1, "Large": 99.96
    #     },
    #     "SSM-au (1.5K)": {
    #         "Small": "-", "Mid": 98.54, "Large": "-"
    #     }
    # }

    # MLRegtest different test types
    data_dict = {
            "RNN (17.1K)": {
                "SR": 94.8, "LR": 85.0, "SA": 71.4, "LA": 66.2
            },
            "GRU (13.4K)": {
                "SR": 97.6, "LR": 96.6, "SA": 84.5, "LA": 84.6
            },
            "LSTM (550K)": {
                "SR": 94.7, "LR": 91.1, "SA": 74.8, "LA": 71.3
            },
            "Transformer (1095K)": {
                "SR": 96.1, "LR": 88.1, "SA": 74.8, "LA": 71.3
            },
            "Mamba (66.6K)": {
                "SR": 99.94, "LR": 98.89, "SA": 94.91, "LA": 93.49
            },
            "SSM-au (49.8K)": {
                "SR": 99.96, "LR": 96.79, "SA": 93.74, "LA": 88.95
            },
            "SSM-au (1.5K)": {
            "SR": 98.54, "LR": 90.37, "SA": 86.41, "LA": 76.55
        },
    }

    print(dict_to_table(data_dict))
