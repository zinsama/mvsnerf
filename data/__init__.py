from .llff import LLFFDataset
from .blender import BlenderDataset
from .dtu_ft import DTU_ft
from .dtu import MVSDatasetDTU
from .owndata import OwnDataset
from .multiset import multiset
from .test import Test
from .myDataset import MyDataset

dataset_dict = {'dtu': MVSDatasetDTU,
                'llff':LLFFDataset,
                'blender': BlenderDataset,
                'dtu_ft': DTU_ft,
                'own': OwnDataset,
                'multiset': multiset,
                'test': Test,
                'MyDataset': MyDataset}