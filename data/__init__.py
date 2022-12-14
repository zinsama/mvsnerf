from .llff import LLFFDataset
from .blender import BlenderDataset
from .dtu_ft import DTU_ft
from .dtu import MVSDatasetDTU
from .owndata import OwnDataset
from .multiset import multiset
from .multiset2 import multiset2
from .test import Test
from .myDataset import MyDataset
from .myDataset2 import MyDataset2

dataset_dict = {'dtu': MVSDatasetDTU,
                'llff':LLFFDataset,
                'blender': BlenderDataset,
                'dtu_ft': DTU_ft,
                'own': OwnDataset,
                'multiset': multiset,
                'multiset2': multiset2,
                'test': Test,
                'MyDataset': MyDataset,
                'MyDataset2': MyDataset2,
                }