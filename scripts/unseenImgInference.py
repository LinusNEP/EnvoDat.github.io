#!/usr/bin/env python
import argparse
import os
import subprocess
import cv2
import matplotlib.pyplot as plt

def run_inference(model_path, source_dir, project, name, conf_threshold):
    command = [
        "yolo", "predict",
        f"model={model_path}",
        f"source={source_dir}",
        f"project={project}",
        f"name={name}",
        f"conf={conf_threshold}",
        "save_txt=True",
        "save_conf=True",
        "exist_ok=True",
    ]
    subprocess.run(command, check=True)

def draw_boxes(image_rgb, labels_path):
    if not (labels_path and os.path.exists(labels_path)):
        return image_rgb

    h, w = image_rgb.shape[:2]
    with open(labels_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue 
            _, x_c, y_c, bw, bh = map(float, parts[:5])
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return image_rgb

def show_image(image_rgb, title=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="YOLO inference + box visualisation on unseen images.")
    parser.add_argument("--model", required=True, help="Path to the trained weights, e.g. best.pt")
    parser.add_argument("--source", required=True, help="Directory of images to run inference on")
    parser.add_argument("--project", default="runs/detect", help="YOLO project (output root) directory")
    parser.add_argument("--name", default="exp", help="Run name under the project directory")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--no-show", action="store_true", help="Skip the matplotlib display loop")
    args = parser.parse_args()

    run_inference(args.model, args.source, args.project, args.name, args.conf)

    labels_dir = os.path.join(args.project, args.name, "labels")

    if args.no_show:
        print(f"Inference complete. Labels saved to: {labels_dir}")
        return

    for image_file in sorted(os.listdir(args.source)):
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image_path = os.path.join(args.source, image_file)
        label_path = os.path.join(labels_dir, os.path.splitext(image_file)[0] + ".txt")

        image = cv2.imread(image_path)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = draw_boxes(image, label_path)
        show_image(image, title=image_file)


if __name__ == "__main__":
    main()
