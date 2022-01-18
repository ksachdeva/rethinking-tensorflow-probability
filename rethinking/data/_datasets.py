import os
import enum
import collections
import pandas as pd

from typing import List
from typing import Dict
from typing import Union
from typing import NamedTuple

import tensorflow as tf

_BASE_URL = "https://raw.githubusercontent.com/rmcelreath/rethinking/Experimental/data"


class RethinkingDataset(enum.Enum):
    Howell1 = "Howell1"
    CherryBlossoms = "cherry_blossoms"
    WaffleDivorce = "WaffleDivorce"
    Milk = "milk"
    Rugged = "rugged"
    Tulips = "tulips"
    Chimpanzees = "chimpanzees"
    UCBadmit = "UCBadmit"
    Kline = "Kline"
    AustinCats = "AustinCats"
    ReedFrogs = "reedfrogs"
    KosterLeckie = "KosterLeckie"
    IslandsDistMatrix = "islandsDistMatrix"
    Primates301 = "Primates301"
    Primates301_vcov_matrix = "Primates301_vcov_matrix"
    Boxes = "Boxes"
    PandaNuts = "Panda_nuts"
    LynxHare = "Lynx_Hare"

    def get_dataset(self) -> pd.DataFrame:
        url = os.path.join(_BASE_URL, self.value + ".csv")
        return pd.read_csv(url, sep=";")


def dataframe_to_tensors(
    name: str,
    df: pd.DataFrame,
    columns: Union[List, Dict],
    default_type=tf.float32,
) -> NamedTuple:
    """name : Name of the dataset
    df : pandas dataframe
    colums : a list of names that have the same type or
             a dictionary where keys are the column names and values are the tensorflow type (e.g. tf.float32)
    """
    if isinstance(columns, dict):
        column_names = columns.keys()
        fields = [tf.cast(df[k].values, dtype=v) for k, v in columns.items()]
    else:
        column_names = columns
        fields = [tf.cast(df[k].values, dtype=default_type) for k in column_names]

    # build the cls
    tuple_cls = collections.namedtuple(name, column_names)
    # build the obj
    return tuple_cls._make(fields)
