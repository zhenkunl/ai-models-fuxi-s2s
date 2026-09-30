# ai-models-fuxi_s2s

`ai-models-fuxi_s2s` is an [ai-models](https://github.com/ecmwf-lab/ai-models) plugin to run [Fudan's FuXi-S2S](https://github.com/tpys/FuXi-S2S.git).

## Installation

To install the package, run:

```bash
pip install ai-models-fuxi-s2s
```

This will install the package and its dependencies, in particular the ONNX runtime. The installation script will attempt to guess which runtime to install. You can force a given runtime by specifying the the `ONNXRUNTIME` variable, e.g.:

```bash
ONNXRUNTIME=onnxruntime-gpu pip install ai-models-fuxi-s2s
```

## Specifying ensemble numbers

There are three ways to control the ensemble members and behaviour of the `FuXi-S2S` `ai-model`.

| Description | Args | Result |
| ----------- | ---- | ------ |
| `type=fc`, single member | `--num-ensemble-members 0` | Will create a `grib` file of `type=fc` |
| N members per process with ID = `range(num-ensemble-members)` | `--num-ensemble-members $N>1` | N ensemble members created all in same process, with id from the range|
| N members per process with controlled ID | `--num-ensemble-members $N>1` `--member-number 1,2...N` | N ensemble members created all in same process, with id controlled from `member-number` |

With these approaches it is possible to create either a single forecast, many ensembles in a single process, or many ensembles over many processes.
