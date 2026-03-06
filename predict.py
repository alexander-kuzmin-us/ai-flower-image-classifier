import argparse
import torch
import json
from PIL import Image
import model_utils

def predict(image_path, model, topk, device):
    image = Image.open(image_path)
    image_np = model_utils.process_image(image)
    image_tensor = torch.from_numpy(image_np).type(torch.FloatTensor).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model.forward(image_tensor)
        
    ps = torch.exp(output)
    top_p, top_indices = ps.topk(topk)
    
    top_p = top_p.cpu().numpy()[0].tolist()
    top_indices = top_indices.cpu().numpy()[0].tolist()
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]
    
    return top_p, top_classes

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('input', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Load model
    model = model_utils.load_checkpoint(args.checkpoint)
    model.to(device)

    # Predict
    probs, classes = predict(args.input, model, args.top_k, device)

    # Map to names if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[c] for c in classes]

    # Print results
    print(f"Predictions for {args.input}:")
    for i in range(len(probs)):
        print(f"{classes[i]}: {probs[i]*100:.2f}%")

if __name__ == '__main__':
    main()