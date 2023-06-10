# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 10:11:13 2022

@author: DELL
"""

import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import transforms
from PIL import Image
from src import u2net_full

model = u2net_full()
for i,a in enumerate(model.children()):
    if i==0:
        print(a)

