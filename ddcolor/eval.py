import os
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from scipy.linalg import sqrtm
from skimage.metrics import peak_signal_noise_ratio
from ddcolor.models.model import DDColor
from ddcolor.models.utils import preprocess_images
from torchvision.models import inception_v3


def load_inception_model():
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = torch.nn.Identity()
    inception_model.eval()
    return inception_model


def preprocess_inception(x):
    x = (x + 1) / 2  # Rescale [-1,1] images to [0,1]
    x = torch.nn.functional.interpolate(
        x, size=(299, 299), mode="bilinear", align_corners=False
    )
    x = (
        x - torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    ) / torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return x


def get_features(images, model):
    images = preprocess_inception(images)
    with torch.no_grad():
        features = model(images)
    return features.cpu().numpy()


def calculate_fid(real_features, generated_features):
    mu_real, sigma_real = real_features.mean(axis=0), np.cov(
        real_features, rowvar=False
    )
    mu_generated, sigma_generated = generated_features.mean(axis=0), np.cov(
        generated_features, rowvar=False
    )

    sum_squared_diff = np.sum((mu_real - mu_generated) ** 2)
    covmean, _ = sqrtm(sigma_real.dot(sigma_generated), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = sum_squared_diff + np.trace(sigma_real + sigma_generated - 2 * covmean)
    return fid


def calculate_colorfulness(image):
    rg = np.std(image[:, :, 0] - image[:, :, 1])
    yb = np.std(image[:, :, 2] - 0.5 * (image[:, :, 0] + image[:, :, 1]))
    std_rgby = np.std([rg, yb])
    mean_rgby = np.mean([rg, yb])
    colorfulness = std_rgby + 0.3 * mean_rgby
    return colorfulness


def evaluate(model_checkpoint_path, test_dir, output_dir):
    if not os.path.exists(model_checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_checkpoint_path}")
    
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found at {test_dir}")
    
    model = DDColor()
    model.load_state_dict(torch.load(model_checkpoint_path))
    model.eval()
    inception_model = load_inception_model()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    test_images = [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.endswith((".jpg", ".png"))
    ]

    real_features = []
    generated_features = []
    colorfulness_scores = []
    psnr_scores = []

    for test_image in tqdm(test_images, desc="Evaluating"):
        image = torchvision.io.read_image(test_image).float() / 255
        image, _, _ = preprocess_images(image.unsqueeze(0))
        with torch.no_grad():
            _, output = model(image)

        real_features.append(get_features(image, inception_model))
        generated_features.append(get_features(output, inception_model))

        output_image = output.squeeze().permute(1, 2, 0).numpy()
        colorfulness_scores.append(calculate_colorfulness(output_image))
        psnr_scores.append(
            peak_signal_noise_ratio(
                image.squeeze().permute(1, 2, 0).numpy(), output_image
            )
        )

    real_features = np.vstack(real_features)
    generated_features = np.vstack(generated_features)

    fid_score = calculate_fid(real_features, generated_features)
    avg_colorfulness = np.mean(colorfulness_scores)
    avg_psnr = np.mean(psnr_scores)

    print(f"Average FID: {fid_score}")
    print(f"Average Colorfulness Score: {avg_colorfulness}")
    print(f"Average PSNR: {avg_psnr}")


if __name__ == "__main__":
    model_checkpoint_path = "ddcolor_checkpoint_8.pth"
    test_dir = "test_images"
    output_dir = "output_images"
    evaluate(model_checkpoint_path, test_dir, output_dir)
