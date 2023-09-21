import argparse
import os
import onnx
import timm
import torch
from onnx_tf.backend import prepare


def export_to_onnx(
    arch,
    imgsz,
    n_classes,
    onnx_path,
    opset_version=11,
    n_channels=3,
    weight_path=None,
    device=None,
):
    if isinstance(imgsz, tuple) or isinstance(imgsz, list):
        assert len(imgsz) <= 2, "List/tuple of sizes must be of format (h, w) or (imgsz,)"
        if len(imgsz) == 2:
            h, w = imgsz[0], imgsz[1]
        else:
            h, w = imgsz[0], imgsz[0]
    else:
        h, w = imgsz, imgsz

    model = timm.create_model(
        model_name=arch,
        num_classes=n_classes
    )
    if weight_path:
        state_dict = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(state_dict)

    if device:
        model.to(device)
        model.eval()

    dummy_input = torch.randn(1, n_channels, h, w)
    dummy_input = dummy_input.to(device)

    dummy_output = model(dummy_input)  # Create graph?
    input_names = ["input"]
    output_names = ["output"]

    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=opset_version
    )


def export_to_saved_model(onnx_model, saved_model_path):
    onnx.checker.check_model(onnx_model)  # Check/validate model
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(saved_model_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        type=str,
        help="Model architecture",
        default=None
    )
    parser.add_argument(
        "--imgsz",
        nargs='*',
        type=int,
        help="Input sizes of model",
        default=None
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        help="Number of classes of output of model",
        default=None
    )
    parser.add_argument(
        "--weight-path",
        type=str,
        help="Path to trained weight of model",
        default=None
    )
    parser.add_argument(
        "--backend",
        type=str,
        help="Backend to export model",
        default='onnx'
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="Path to save exported model",
        default=None
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.backend.lower() == 'onnx':
        if args.save_path is None:
            args.save_path = f"{args.arch}.onnx"
        else:
            os.makedirs(args.save_path, exist_ok=True)
            args.save_path = os.path.join(args.save_path, f"{args.arch}.onnx")

        export_to_onnx(
            arch=args.arch,
            imgsz=args.imgsz,
            n_classes=args.n_classes,
            n_channels=3,
            weight_path=args.weight_path,
            onnx_path=args.save_path
        )

    elif args.backend.lower() == 'tf':
        if args.save_path is None:
            args.save_path = f"{args.arch}_saved_model"
            onnx_path = f"{args.arch}.onnx"
        else:
            os.makedirs(args.save_path, exist_ok=True)
            onnx_path = os.path.join(args.save_path, f"{args.arch}.onnx")
            args.save_path = os.path.join(args.save_path, f"{args.arch}_saved_model")

        export_to_onnx(
            arch=args.arch,
            imgsz=args.imgsz,
            n_classes=args.n_classes,
            n_channels=3,
            weight_path=args.weight_path,
            onnx_path=onnx_path
        )
        onnx_model = onnx.load(onnx_path)

        export_to_saved_model(onnx_model, saved_model_path=args.save_path)
