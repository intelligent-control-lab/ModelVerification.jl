# NNet Converter

The following Python scripts are used to convert between different neural network file formats. The supported file formats are as follows:
- [`.onnx` (Open Neural Network Exchange)](https://onnx.ai/): Specification that defines how models should be constructed and the operators in the graph. Open-source project under the Linux Foundation. 
- [`.pb` (protobug)](https://github.com/protocolbuffers/protobuf): Used by TensorFlow's serving when the model needs to be deployed for production. Open-source project that is currently overviewd by Google.
- [`.h5` (HDF5 binary data format)](https://en.wikipedia.org/wiki/Hierarchical_Data_Format): Originally used by Keras to save models. This file format is less general and more "data-oriented" and less programmatic than `.pb`, but simpler to use than `.pb`. It is easily convertible to `.pb`.
- [`.nnet` (NNet)](https://github.com/sisl/NNet): Developed by the [Stanford Intelligent Systems Laboratory](https://sisl.stanford.edu/), initially to define aircraft collision avoidance neural networks in human-readable text document. This format is a simple text-based format for feed-forward, fully-connected, ReLU-activate neural networks.
- [`.pt` (PyTorch)](https://pytorch.org/tutorials/beginner/saving_loading_models.html): Used by PyTorch.


## [H5 to ONNX](https://github.com/intelligent-control-lab/ModelVerification.jl/blob/master/NNet/converters/h52onnx.py)
Converts a `.h5` model to an `.onnx` model.
```console
~/ModelVerification.jl/NNet/converters$ python h52onnx.py --model_path "[path/to/h5/file]" --name_model "[path/to/converted/onnx/file]" --test_conversion [True/False]
```

## [NNET to ONNX](https://github.com/intelligent-control-lab/ModelVerification.jl/blob/master/NNet/converters/nnet2onnx.py)
Converts a `.nnet` model to an `.onnx` model.
```console
~/ModelVerification.jl/NNet/converters$ python nnet2onnx.py [nnetFile] [onnxFile] [outputName] [normalizeNetwork]
```
where 
- `nnetFile`: (string) .nnet file to convert to onnx.
- `onnxFile`: (string, optional) Optional, name for the created .onnx file.
- `outputName`: (string, optional) Optional, name of the output variable in onnx.
- `normalizeNetwork`: (bool, optional) If true, adapt the network weights and biases so that networks and inputs do not need to be normalized. Default is `False`.

## [NNET to PB](https://github.com/intelligent-control-lab/ModelVerification.jl/blob/master/NNet/converters/nnet2pb.py)
Converts a `.nnet` model to a `.pb` model.
```console
~/ModelVerification.jl/NNet/converters$ python nnet2pb.py [nnetFile] [pbFile] [output_node_names]
```
- `nnetFile` (string): A .nnet file to convert to Tensorflow format.
- `pbFile` (string, optional): Name for the created `.pb` file. Default: `""`.
- `output_node_names` (string, optional): Name of the final operation in the Tensorflow graph. Default: `"y_out"`.

## [ONNX to NNET](https://github.com/intelligent-control-lab/ModelVerification.jl/blob/master/NNet/converters/onnx2nnet.py)
Converts an `.onnx` model to a `.nnet` model.
```console
~/ModelVerification.jl/NNet/converters$ python onnx2nnet.py [onnxFile] [nnetFile]
```
- `onnxFile` (string): Path to `.onnx` file.
- `nnetFile` (string, optional): Name for the created `.nnet` file.

## [PB to NNET](https://github.com/intelligent-control-lab/ModelVerification.jl/blob/master/NNet/converters/pb2nnet.py)
Converts a `.pb` model to a `.nnet` model.
```console
~/ModelVerification.jl/NNet/converters$ python pb2nnet.py [pbFile]
```
- `pbFile` (string): If `savedModel` is false, it is the path to the frozen graph `.pb` file. If `savedModel` is true, it is the path to the `savedModel` folder, which contains `.pb` file and variables subdirectory.

## [PT to ONNX](https://github.com/intelligent-control-lab/ModelVerification.jl/blob/master/NNet/converters/pt2onnx.py)
Converts a `.pt` model to an `.onnx` model.
```console
~/ModelVerification.jl/NNet/converters$ python pt2onnx.py --model_path "[path/to/h5/file]" --name_model "[path/to/converted/onnx/file]" --test_conversion [True/False]
```