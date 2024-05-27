from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image, ImageDraw, ImageFilter
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def load_image(image_path):
    img = Image.open(image_path)
    img_cropped, prob = mtcnn(img, return_prob=True)
    if img_cropped is not None and prob > 0.9:
        print(f'Face detected in {image_path} with probability: {prob}')
        return img_cropped
    else:
        print(f'No face detected in {image_path} or low probability.')
        return None

def get_embedding(image_tensor):
    aligned = image_tensor.unsqueeze(0).to(device)
    embedding = resnet(aligned).detach().cpu()
    return embedding

def extract_faces(image_path):
    img = Image.open(image_path)
    img_cropped_list, probs = mtcnn(img, return_prob=True)

    if not isinstance(img_cropped_list, list):
        img_cropped_list = [img_cropped_list]
        probs = [probs]

    faces = []
    for img_cropped, prob in zip(img_cropped_list, probs):
        if prob > 0.9:
            print(f'Face detected with probability: {prob}')
            faces.append(img_cropped)
    return faces, img

def find_most_similar_face(img1_path, img2_path):
    try:
        img1_folder_name = os.path.basename(os.path.dirname(img1_path))
        img1_tensor = load_image(img1_path)
        if img1_tensor is None:
            return None

        img1_embedding = get_embedding(img1_tensor)

        faces_in_img2, original_img2 = extract_faces(img2_path)
        if not faces_in_img2:
            return None

        img2_embeddings = []
        for face in faces_in_img2:
            img2_embedding = get_embedding(face)
            img2_embeddings.append(img2_embedding)

        img2_embeddings = torch.cat(img2_embeddings)
        distances = torch.norm(img2_embeddings - img1_embedding, dim=1)
        min_dist, min_idx = torch.min(distances, dim=0)
        most_similar_face_idx = min_idx.item()
        most_similar_distance = min_dist.item()

        print(f'The most similar face to img1 is face number: {most_similar_face_idx} in img2 with distance: {most_similar_distance}')

        boxes, probs = mtcnn.detect(original_img2)
        if boxes is None:
            return None

        blurred_img = original_img2.filter(ImageFilter.GaussianBlur(15))
        draw = ImageDraw.Draw(original_img2)

        for i, box in enumerate(boxes):
            if i == most_similar_face_idx:
                box = [int(b) for b in box]
                expansion_factor = 0.2
                box_width = box[2] - box[0]
                box_height = box[3] - box[1]
                box[0] -= int(box_width * expansion_factor)
                box[1] -= int(box_height * expansion_factor)
                box[2] += int(box_width * expansion_factor)
                box[3] += int(box_height * expansion_factor)

                draw.rectangle(box, outline='red', width=2)
                region = original_img2.crop((box[0], box[1], box[2], box[3]))
                blurred_img.paste(region, (box[0], box[1]))

        return blurred_img

    except Exception as e:
        print(f'Error: {e}')
        return None
