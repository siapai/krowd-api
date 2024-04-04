import time
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from torchvision.models.detection import FasterRCNN
import torch
import numpy as np

INPUT_SIZE = (480, 640)


def predict(image_name: str, model: FasterRCNN, device: torch.device):
    try:
        image_path = f'files/{image_name}'
        image = Image.open(image_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
        ])
        input_tensor = transform(image).unsqueeze(0)

        model.eval()
        start_time = time.time()
        with torch.inference_mode():
            predictions = model(input_tensor.to(device))
        end_time = time.time()
        latency = end_time - start_time
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in predictions]
        boxes = outputs[0]['boxes']
        scores = outputs[0]['scores']
        min_score = 0.8

        keep = [i for i, score in enumerate(scores) if score > min_score]

        filtered_boxes = [boxes[idx] for idx in keep]
        filtered_scores = [scores[idx] for idx in keep]

        output_name = f"{image_name.split('.')[0]}_pth.{image_name.split('.')[-1]}"
        output_path = f"files/{output_name}"

        image_np = input_tensor.squeeze().cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))
        image_np = (image_np * 255).astype(np.uint8)

        draw_boxes(image_np, filtered_boxes, [f'{score.item():.4f}' for score in filtered_scores], output_path)

        output_static = f"static/{output_name}"
        return filtered_boxes, filtered_scores, latency, output_static

    except Exception as e:
        print(e)


def predict_onnx(image_name, session):
    try:
        image_path = f'files/{image_name}'
        image = Image.open(image_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
        ])

        input_tensor = transform(image).unsqueeze(0)

        # Run inference
        inputs = {session.get_inputs()[0].name: input_tensor.detach().cpu().numpy()}

        start_time = time.time()
        outputs = session.run(None, inputs)
        end_time = time.time()
        latency = end_time - start_time

        boxes, labels, scores = outputs[0], outputs[1], outputs[2]

        min_score = 0.8
        keep = [i for i, score in enumerate(scores) if score > min_score]

        filtered_boxes = [boxes[idx] for idx in keep]
        filtered_scores = [scores[idx] for idx in keep]

        output_name = f"{image_name.split('.')[0]}_onnx.{image_name.split('.')[-1]}"
        output_path = f"files/{output_name}"

        image_np = input_tensor.squeeze().cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))
        image_np = (image_np * 255).astype(np.uint8)

        draw_boxes(image_np, filtered_boxes, [f'{score.item():.4f}' for score in filtered_scores], output_path)

        output_static = f"static/{output_name}"

        return filtered_boxes, filtered_scores, latency, output_static

    except Exception as e:
        print(e)


def draw_boxes(image, boxes, labels, output_path):
    image = Image.fromarray(image)
    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Define the font for the labels
    font = ImageFont.load_default()

    # Define the colors for bounding boxes and labels
    box_color = (255, 0, 0)  # Red color
    label_color = (255, 255, 0)  # Yellow color

    # Draw bounding boxes and labels on the image
    for box, label in zip(boxes, labels):
        # Convert box coordinates to integers
        box = [int(coord) for coord in box]

        # Draw the bounding box
        draw.rectangle(box, outline=box_color)

        # Display the label
        draw.text((box[0], box[1] - 10), label, fill=label_color, font=font)

    # Save the image with bounding boxes and labels
    image.save(output_path)
