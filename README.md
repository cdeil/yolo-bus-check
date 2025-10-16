# yolo-bus-check

Check object detection results on bus image with Ultralytics YOLO11n
showing that there's a bug in the CoreML exporter.

In `make_models.py` we download the `yolo11n.pt` model
and export it to `yolo11n.mlpackage`.

In `model_predict_pt.py` we predict and show results in PT,
which has correct result of 94% confidence for the bus.

The `results.coreml.png` screenshot shows that the CoreML model
gives incorrect confidences (always 100%).

This depends a bit on what PyTorch/CoreMLTools/ultralytics version,
but I never found any version combination giving "correct" results
which I presume the PT results are.
