import os
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from scipy.linalg import sqrtm
from skimage.metrics import peak_signal_noise_ratio
from models.model import DDColor
from models.utils import preprocess_images
from torchvision.models import inception_v3


# Load pre-trained Inception model
def load_inception_model():
    # Load pre-trained Inception v3 model
    inception_model = inception_v3(pretrained=True)
    inception_model.eval()

    # Preprocess function for Inception model
    def preprocess(x):
        x = x * 2 - 1  # Normalize to [-1, 1]
        x = x.permute(0, 3, 1, 2)  # NHWC to NCHW
        return x

    # Define the feature extraction function
    def get_features(x):
        x = preprocess(x)
        features = inception_model(x)
        features = features.detach()
        return features

    return get_features


def calculate_fid(real_images, generated_images, inception_model):
    real_features = inception_model(real_images)
    generated_features = inception_model(generated_images)

    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(
        real_features, rowvar=False
    )
    mu_generated, sigma_generated = np.mean(generated_features, axis=0), np.cov(
        generated_features, rowvar=False
    )

    sum_squared = np.sum((mu_real - mu_generated) ** 2.0)
    covmean = sqrtm(sigma_real.dot(sigma_generated))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = sum_squared + np.trace(sigma_real + sigma_generated - 2.0 * covmean)
    return fid


def calculate_colorfulness(images):
    colorfulness_scores = []
    for image in images:
        rg = np.sqrt(np.mean(np.square(image[:, :, 0] - image[:, :, 1])))
        yb = np.sqrt(
            np.mean(np.square(image[:, :, 2] - np.mean(image[:, :, 0:2], axis=2)))
        )
        colorfulness_scores.append(np.sqrt(rg**2 + yb**2) + 0.3 * np.mean(image))

    return np.mean(colorfulness_scores)


def calculate_psnr(real_images, generated_images):
    psnr_scores = []
    for real_image, generated_image in zip(real_images, generated_images):
        psnr_scores.append(peak_signal_noise_ratio(real_image, generated_image))

    return np.mean(psnr_scores)


def evaluate(model, test_dir, output_dir):
    inception_model = load_inception_model()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    test_images = [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if f.endswith((".jpg", ".png"))
    ]

    fid_scores = []
    cf_scores = []
    psnr_scores = []

    for test_image in tqdm(test_images, desc="Evaluating"):
        image = torchvision.io.read_image(test_image).unsqueeze(0)
        image, _, _ = preprocess_images(image)

        output, colored_images = model(image)

        assert output.shape == (
            1,
            2,
            256,
            256,
        ), f"Output shape is incorrect: {output.shape}"

        assert (
            len(colored_images) == 1
        ), f"Incorrect number of colorized images: {len(colored_images)}"
        colored_image = colored_images[0, ...].permute(1, 2, 0).detach() * 255
        colored_image = colored_image.numpy().astype(np.uint8)
        assert colored_image.shape == (
            256,
            256,
            3,
        ), f"Colorized image shape is incorrect: {colored_image.shape}"

        filename = os.path.basename(test_image)
        output_path = os.path.join(output_dir, filename)
        torchvision.io.write_png(
            torch.tensor(colored_image).permute(2, 0, 1), output_path
        )

        # Calculate evaluation metrics
        real_image = torchvision.io.read_image(test_image).permute(1, 2, 0).numpy()
        fid_scores.append(
            calculate_fid(
                np.expand_dims(real_image, 0),
                np.expand_dims(colored_image, 0),
                inception_model,
            )
        )
        cf_scores.append(calculate_colorfulness(np.expand_dims(colored_image, 0)))
        psnr_scores.append(
            calculate_psnr(
                np.expand_dims(real_image, 0), np.expand_dims(colored_image, 0)
            )
        )

    # Print average evaluation metrics
    print(f"Average FID: {np.mean(fid_scores)}")
    print(f"Average Colorfulness Score: {np.mean(cf_scores)}")
    print(f"Average PSNR: {np.mean(psnr_scores)}")


if __name__ == "__main__":
    model = DDColor()
    test_dir = "test_images"
    output_dir = "colorized_images"

    evaluate(model, test_dir, output_dir)
