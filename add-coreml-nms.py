# Copyright (C) 2021 DB Systel GmbH.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to add non maximum suppression to an existing CoreML YOLO model.

Modified from the script at src/coreml_export/main.py in the project
"yolov5 - CoreML Tools" on GitHub by DB Systel GmbH, found at:
    https://github.com/dbsystel/yolov5-coreml-tools
"""

import argparse

import coremltools as ct
import torch


# The labels of your model
classLabels = ["Longitudinal Crack", "Transverse Crack", "Alligator Crack", "Potholes"]
numberOfClassLabels = len(classLabels)
outputSize = numberOfClassLabels + 5

#  Attention: Some models are reversed!
reverseModel = False

strides = [8, 16, 32]
if reverseModel:
    strides.reverse()
featureMapDimensions = [640 // stride for stride in strides]

anchors = ([ 12,  16,  19,  36,  40,  28],
           [ 36,  75,  76,  55,  72, 146],
           [142, 110, 192, 243, 459, 401])  # Take these from the <model>.yml
if reverseModel:
    anchors = anchors[::-1]

anchorGrid = torch.tensor(anchors).float().view(3, -1, 1, 1, 2)


def make_grid(nx, ny):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
    return torch.stack((xv, yv), 2).view((ny, nx, 2)).float()


def addExportLayerToCoreML(builder):
    """
    Adds the YOLOv5 export layer to the CoreML model.
    """
    outputNames = [output.name for output in builder.spec.description.output]

    for i, outputName in enumerate(outputNames):
        # formulas: https://github.com/ultralytics/yolov5/issues/471
        builder.add_activation(name=f"sigmoid_{outputName}", non_linearity="SIGMOID",
                               input_name=outputName, output_name=f"{outputName}_sigmoid")

        ### Coordinates calculation ###
        # input (1, 3, nC, nC, 85), output (1, 3, nC, nC, 2) -> nC = 640 / strides[i]
        builder.add_slice(name=f"slice_coordinates_xy_{outputName}", input_name=f"{outputName}_sigmoid",
                          output_name=f"{outputName}_sliced_coordinates_xy", axis="width", start_index=0, end_index=2)
        # x,y * 2
        builder.add_elementwise(name=f"multiply_xy_by_two_{outputName}", input_names=[
                                f"{outputName}_sliced_coordinates_xy"], output_name=f"{outputName}_multiplied_xy_by_two", mode="MULTIPLY", alpha=2)
        # x,y * 2 - 0.5
        builder.add_elementwise(name=f"subtract_0_5_from_xy_{outputName}", input_names=[
                                f"{outputName}_multiplied_xy_by_two"], output_name=f"{outputName}_subtracted_0_5_from_xy", mode="ADD", alpha=-0.5)
        grid = make_grid(
            featureMapDimensions[i], featureMapDimensions[i]).numpy()
        # x,y * 2 - 0.5 + grid[i]
        builder.add_bias(name=f"add_grid_from_xy_{outputName}", input_name=f"{outputName}_subtracted_0_5_from_xy",
                         output_name=f"{outputName}_added_grid_xy", b=grid, shape_bias=grid.shape)
        # (x,y * 2 - 0.5 + grid[i]) * stride[i]
        builder.add_elementwise(name=f"multiply_xy_by_stride_{outputName}", input_names=[
                                f"{outputName}_added_grid_xy"], output_name=f"{outputName}_calculated_xy", mode="MULTIPLY", alpha=strides[i])

        # input (1, 3, nC, nC, 85), output (1, 3, nC, nC, 2)
        builder.add_slice(name=f"slice_coordinates_wh_{outputName}", input_name=f"{outputName}_sigmoid",
                          output_name=f"{outputName}_sliced_coordinates_wh", axis="width", start_index=2, end_index=4)
        # w,h * 2
        builder.add_elementwise(name=f"multiply_wh_by_two_{outputName}", input_names=[
                                f"{outputName}_sliced_coordinates_wh"], output_name=f"{outputName}_multiplied_wh_by_two", mode="MULTIPLY", alpha=2)
        # (w,h * 2) ** 2
        builder.add_unary(name=f"power_wh_{outputName}", input_name=f"{outputName}_multiplied_wh_by_two",
                          output_name=f"{outputName}_power_wh", mode="power", alpha=2)
        # (w,h * 2) ** 2 * anchor_grid[i]
        anchor = anchorGrid[i].expand(-1, featureMapDimensions[i],
                                      featureMapDimensions[i], -1).numpy()
        builder.add_load_constant_nd(
            name=f"anchors_{outputName}", output_name=f"{outputName}_anchors", constant_value=anchor, shape=anchor.shape)
        builder.add_elementwise(name=f"multiply_wh_with_achors_{outputName}", input_names=[
                                f"{outputName}_power_wh", f"{outputName}_anchors"], output_name=f"{outputName}_calculated_wh", mode="MULTIPLY")

        builder.add_concat_nd(name=f"concat_coordinates_{outputName}", input_names=[
                              f"{outputName}_calculated_xy", f"{outputName}_calculated_wh"], output_name=f"{outputName}_raw_coordinates", axis=-1)
        builder.add_scale(name=f"normalize_coordinates_{outputName}", input_name=f"{outputName}_raw_coordinates",
                          output_name=f"{outputName}_raw_normalized_coordinates", W=torch.tensor([1 / 640]).numpy(), b=0, has_bias=False)

        ### Confidence calculation ###
        builder.add_slice(name=f"slice_object_confidence_{outputName}", input_name=f"{outputName}_sigmoid",
                          output_name=f"{outputName}_object_confidence", axis="width", start_index=4, end_index=5)
        builder.add_slice(name=f"slice_label_confidence_{outputName}", input_name=f"{outputName}_sigmoid",
                          output_name=f"{outputName}_label_confidence", axis="width", start_index=5, end_index=0)
        # confidence = object_confidence * label_confidence
        builder.add_multiply_broadcastable(name=f"multiply_object_label_confidence_{outputName}", input_names=[
                                           f"{outputName}_label_confidence", f"{outputName}_object_confidence"], output_name=f"{outputName}_raw_confidence")

        # input: (1, 3, nC, nC, 85), output: (3 * nc^2, 85)
        builder.add_flatten_to_2d(
            name=f"flatten_confidence_{outputName}", input_name=f"{outputName}_raw_confidence", output_name=f"{outputName}_flatten_raw_confidence", axis=-1)
        builder.add_flatten_to_2d(
            name=f"flatten_coordinates_{outputName}", input_name=f"{outputName}_raw_normalized_coordinates", output_name=f"{outputName}_flatten_raw_coordinates", axis=-1)

    builder.add_concat_nd(name="concat_confidence", input_names=[
                          f"{outputName}_flatten_raw_confidence" for outputName in outputNames], output_name="raw_confidence", axis=-2)
    builder.add_concat_nd(name="concat_coordinates", input_names=[
                          f"{outputName}_flatten_raw_coordinates" for outputName in outputNames], output_name="raw_coordinates", axis=-2)

    builder.set_output(output_names=["raw_confidence", "raw_coordinates"], output_dims=[
                       (25200, numberOfClassLabels), (25200, 4)])


def createNmsModelSpec(nnSpec):
    """
    Create a CoreML model with NMS to filter the results of the model.
    """
    nmsSpec = ct.proto.Model_pb2.Model()
    nmsSpec.specificationVersion = 4

    # Define input and outputs of the model
    for i in range(2):
        nnOutput = nnSpec.description.output[i].SerializeToString()

        nmsSpec.description.input.add()
        nmsSpec.description.input[i].ParseFromString(nnOutput)

        nmsSpec.description.output.add()
        nmsSpec.description.output[i].ParseFromString(nnOutput)

    nmsSpec.description.output[0].name = "confidence"
    nmsSpec.description.output[1].name = "coordinates"

    # Define output shape of the model
    outputSizes = [numberOfClassLabels, 4]
    for i in range(len(outputSizes)):
        maType = nmsSpec.description.output[i].type.multiArrayType
        # First dimension of both output is the number of boxes, which should be flexible
        maType.shapeRange.sizeRanges.add()
        maType.shapeRange.sizeRanges[0].lowerBound = 0
        maType.shapeRange.sizeRanges[0].upperBound = -1
        # Second dimension is fixed, for "confidence" it"s the number of classes, for coordinates it"s position (x, y) and size (w, h)
        maType.shapeRange.sizeRanges.add()
        maType.shapeRange.sizeRanges[1].lowerBound = outputSizes[i]
        maType.shapeRange.sizeRanges[1].upperBound = outputSizes[i]
        del maType.shape[:]

    # Define the model type non maximum supression
    nms = nmsSpec.nonMaximumSuppression
    nms.confidenceInputFeatureName = "raw_confidence"
    nms.coordinatesInputFeatureName = "raw_coordinates"
    nms.confidenceOutputFeatureName = "confidence"
    nms.coordinatesOutputFeatureName = "coordinates"
    nms.iouThresholdInputFeatureName = "iouThreshold"
    nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
    # Some good default values for the two additional inputs, can be overwritten when using the model
    nms.iouThreshold = 0.6
    nms.confidenceThreshold = 0.4
    nms.stringClassLabels.vector.extend(classLabels)

    return nmsSpec


def combineModelsAndExport(builderSpec, nmsSpec, fileName, quantize=False):
    """
    Combines the CoreML model with export logic and the NMS to one final model.
    Optionally save with different quantization (32, 16, 8) (Works only if on
    macOS).
    """
    try:
        print(f"Combine CoreML model with NMS and export model")
        # Combine models to a single one
        pipeline = ct.models.pipeline.Pipeline(input_features=[
            ("image", ct.models.datatypes.Array(3, 460, 460)),
            ("iouThreshold", ct.models.datatypes.Double()),
            ("confidenceThreshold", ct.models.datatypes.Double()),
        ], output_features=["confidence", "coordinates"])

        # Required version (>= iOS 13) in order for NMS to work
        pipeline.spec.specificationVersion = 4

        pipeline.add_model(builderSpec)
        pipeline.add_model(nmsSpec)

        pipeline.spec.description.input[0].ParseFromString(
            builderSpec.description.input[0].SerializeToString())
        pipeline.spec.description.output[0].ParseFromString(
            nmsSpec.description.output[0].SerializeToString())
        pipeline.spec.description.output[1].ParseFromString(
            nmsSpec.description.output[1].SerializeToString())

        # Metadata for the model‚
        pipeline.spec.description.input[
            1].shortDescription = "(optional) IOU Threshold override (Default: 0.6)"
        pipeline.spec.description.input[
            2].shortDescription = "(optional) Confidence Threshold override (Default: 0.4)"
        pipeline.spec.description.output[0].shortDescription = u"Boxes \xd7 Class confidence"
        pipeline.spec.description.output[
            1].shortDescription = u"Boxes \xd7 [x, y, width, height] (relative to image size)"
        pipeline.spec.description.metadata.versionString = "1.0"
        pipeline.spec.description.metadata.shortDescription = "YOLOv7-RDD"
        pipeline.spec.description.metadata.author = "Vegard Skui & Victor Pierre Hamran"
        pipeline.spec.description.metadata.license = ""

        model = ct.models.MLModel(pipeline.spec)
        model.save(fileName)

        if quantize:
            fileName16 = fileName.replace(".mlmodel", "_16.mlmodel")
            modelFp16 = ct.models.neural_network.quantization_utils.quantize_weights(
                model, nbits=16)
            modelFp16.save(fileName16)

            fileName8 = fileName.replace(".mlmodel", "_8.mlmodel")
            modelFp8 = ct.models.neural_network.quantization_utils.quantize_weights(
                model, nbits=8)
            modelFp8.save(fileName8)

        print(f"CoreML export success, saved as {fileName}")
    except Exception as e:
        print(f"CoreML export failure: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    # Load the existing CoreML model without NMS
    mlmodel = ct.models.MLModel(args.input)

    # Create a neural network builder based of the existing specification
    spec = mlmodel.get_spec()
    builder = ct.models.neural_network.NeuralNetworkBuilder(spec=spec)

    # Add export logic to CoreML model
    addExportLayerToCoreML(builder)

    # Create NMS logic
    nmsSpec = createNmsModelSpec(builder.spec)

    # Combine model with export logic and NMS logic
    combineModelsAndExport(builder.spec, nmsSpec, args.output, args.quantize)


if __name__ == "__main__":
    main()
