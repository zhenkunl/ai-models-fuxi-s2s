# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import logging
import os

import numpy as np
import onnxruntime as ort
from ai_models.model import Model
import pandas as pd
import xarray as xr
from copy import deepcopy

LOG = logging.getLogger(__name__)


class FuXi_S2S(Model):

    grib_edition = 1
    grib_extra_metadata = {"type": "pf"}

    download_url = "https://zenodo.org/records/15718402/files/{file}"
    download_files = ["fuxi_s2s.onnx", "fuxi_s2s"]

    # Input
    area = [90, 0, -90, 358.5]
    grid = [1.5, 1.5]

    param_sfc = ["2t", "2d", "sst", "ttr", "10u", "10v", "100u", "100v", "msl", "tcwv", "tp"]

    param_level_pl = (
        ["z", "t", "u", "v", "q"],
        [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
    )

    # Output
    expver = "fs2s"

    def __init__(self, num_threads=1, **kwargs):
        super().__init__(**kwargs)
        self.num_threads = num_threads
        self.hour_steps = 24
        self.lagged = [-24, 0]

        self.ordering = [
            f"{param}{level}"
            for param in self.param_level_pl[0]
            for level in self.param_level_pl[1]
        ] + self.param_sfc

        if isinstance(self.member_number, str):
            self.member_number = list(set(map(int, self.member_number.split(","))))
        elif isinstance(self.member_number, int):
            self.member_number = [int(self.member_number)]
        elif self.member_number is None:
            self.member_number = list(range(1, self.num_ensemble_members + 1))
        else:
            raise TypeError(f"`member_number` must be a string or int, not {type(self.member_number)}")

        if not len(self.member_number) == self.num_ensemble_members:
            raise ValueError(
                f"Number of ensemble members must match `member_number`,\nNot {self.num_ensemble_members=} and {self.member_number=}"
            )

        if "stream" not in getattr(self, "metadata", {}):
            self.grib_extra_metadata["stream"] = "enfo"

    def get_input(self):
        fields_pl = self.fields_pl
        param, level = self.param_level_pl
        fields_pl = fields_pl.sel(param=param, level=level)
        fields_pl = fields_pl.order_by('valid_datetime', param=param, level=level)
        fields_pl_numpy = fields_pl.to_numpy(dtype=np.float32)
        fields_pl_numpy1, fields_pl_numpy2 = np.split(fields_pl_numpy, 2, axis=0)

        fields_sfc = self.fields_sfc
        fields_sfc = fields_sfc.sel(param=self.param_sfc)
        fields_sfc = fields_sfc.order_by('valid_datetime', param=self.param_sfc)
        fields_sfc_numpy = fields_sfc.to_numpy(dtype=np.float32)
        fields_sfc_numpy1, fields_sfc_numpy2 = np.split(fields_sfc_numpy, 2, axis=0)

        prev = np.concatenate([fields_pl_numpy1, fields_sfc_numpy1], axis=0)
        curr = np.concatenate([fields_pl_numpy2, fields_sfc_numpy2], axis=0)
        input = np.stack([prev, curr], axis=0)

        self.template = fields_pl[len(fields_pl) // len(self.lagged) :] + fields_sfc[len(fields_sfc) // len(self.lagged) :]

        return input

    def load_model(self):
        ort.set_default_logger_severity(3)
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena=False
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False

        # if self.device == "cuda":
        #     providers = [('CUDAExecutionProvider', {'arena_extend_strategy':'kSameAsRequested'})]
        # elif self.device == "cpu":
        #     providers=['CPUExecutionProvider']
        #     options.intra_op_num_threads = 24
        # else:
        #     raise ValueError("device must be cpu or cuda!")

        model_file = os.path.join(self.assets, self.download_files[0])

        model = ort.InferenceSession(
            model_file,
            sess_options=options,
            providers=self.providers
        )
        return model

    def run(self):

        oper_fcst: bool = False
        if self.num_ensemble_members == 0:
            oper_fcst = True
            # Set the number of ensemble members to 1, and id to 0.
            self.num_ensemble_members = 1
            self.member_number = [0]
            self.grib_extra_metadata = {"type": "fc", "stream": "oper"}

        model = self.load_model()
        input = self.get_input()

        batch = input[None]

        for member in range(self.num_ensemble_members):
            if oper_fcst:
                extra_write_kwargs = {}
            else:
                extra_write_kwargs = dict(number=self.member_number[member])

            new_input = deepcopy(batch)
            with self.stepper(24) as stepper:
                for i in range(self.lead_time // self.hour_steps):
                    step = (i + 1) * self.hour_steps

                    inputs = {'input': new_input}
                    inputs['step'] = np.array([i], dtype=np.float32)
                    new_input, = model.run(None, inputs)
                    output = deepcopy(new_input[:, -1:])
                    # breakpoint()

                    for k, fs in enumerate(self.template):
                        self.write(
                            output[0, 0, k, ...], template=fs, step=step, **extra_write_kwargs,
                        )

                    stepper(i, step)


    def parse_model_args(self, args):
        import argparse

        parser = argparse.ArgumentParser("ai-models fuxi_s2s")
        parser.add_argument(
            "--num-ensemble-members",
            type=int,
            help="Number of ensemble members to run, If 0 set data as 'type=fc'.",
            default=0,
        )
        parser.add_argument(
            "--member-number",
            help="Member Number/s, if multiple num-ensemble-members>1, seperate by ','. If not given will be range.",
            default=None,
        )
        parser.add_argument("--use-an", action="store_true")
        parser.add_argument("--override-constants")
        return parser.parse_args(args)
