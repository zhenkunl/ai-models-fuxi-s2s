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