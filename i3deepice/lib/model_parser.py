#!/usr/bin/env python
# coding: utf-8

"""This file is part of DeepIceLearning
DeepIceLearning is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import importlib
import os
import sys

def parse_functional_model(cfg_file, exp_file, only_model=False):
    sys.path.append(os.path.dirname(cfg_file))
    sys.path.append(os.getcwd()+"/"+os.path.dirname(cfg_file))
    mname = os.path.splitext(os.path.basename(cfg_file))[0]
    func_model_def = importlib.import_module(mname)
    sys.path.pop()
    if only_model:
        return func_model_def
    inputs = func_model_def.inputs
    outputs = func_model_def.outputs
    loss_dict = {}
    if hasattr(func_model_def, 'loss_weights'):
        loss_dict['loss_weights'] = func_model_def.loss_weights
    if hasattr(func_model_def, 'loss_functions'):
        loss_dict['loss'] = func_model_def.loss_functions
    if hasattr(func_model_def, 'metrics'):
        loss_dict['metrics'] = func_model_def.metrics
    if hasattr(func_model_def, 'mask'):
        mask_func = func_model_def.mask
    else:
        mask_func = None
    in_shapes, in_trans, out_shapes, out_trans = \
        prepare_io_shapes(inputs, outputs, exp_file)
    print('----In  Shapes-----\n {}'.format(in_shapes))
    print('----Out Shapes----- \n {}'.format(out_shapes))
    print('--- Loss Settings ---- \n {}'.format(loss_dict))
    model = func_model_def.model(in_shapes, out_shapes)
    return model, in_shapes, in_trans, out_shapes, out_trans, loss_dict, mask_func


