#!/usr/bin/env python
import argparse
import sys

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("Missing dependency. Install with:  pip install ultralytics")

def main():
    ap = argparse.ArgumentParser(description="Train a YOLO detector on EnvoDat.")
    ap.add_argument("--data", required=True, help="Path to the dataset YAML (envodata-*.yaml)")
    ap.add_argument("--model", default="yolo11n.pt", help="Pretrained weights or model name")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    ap.add_argument("--lrf", type=float, default=0.1, help="Final learning-rate fraction")
    ap.add_argument("--momentum", type=float, default=0.937)
    ap.add_argument("--weight-decay", type=float, default=0.0005)
    ap.add_argument("--patience", type=int, default=50, help="Early-stopping patience (epochs)")
    ap.add_argument("--save-period", type=int, default=10, help="Checkpoint every N epochs (-1 to disable)")
    ap.add_argument("--device", default=None, help="cuda device id(s), 'cpu', or leave unset to auto-detect")
    ap.add_argument("--name", default=None, help="Run name (defaults to Ultralytics auto-naming)")
    ap.add_argument("--val-test", action="store_true", help="Also validate on the test split after training")
    ap.add_argument("--export", default=None,
                    help="Export format after training, e.g. onnx, torchscript, engine")
    args = ap.parse_args()

    model = YOLO(args.model)
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        patience=args.patience,
        save_period=args.save_period,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.name:
        train_kwargs["name"] = args.name

    model.train(**train_kwargs)

    val_metrics = model.val(split="val")
    print(f"[val]  mAP50={val_metrics.box.map50:.4f}  mAP50-95={val_metrics.box.map:.4f}")

    if args.val_test:
        test_metrics = model.val(split="test")
        print(f"[test] mAP50={test_metrics.box.map50:.4f}  mAP50-95={test_metrics.box.map:.4f}")

    if args.export:
        path = model.export(format=args.export)
        print(f"Exported model to: {path}")


if __name__ == "__main__":
    main()
