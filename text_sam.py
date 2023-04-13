import os
import cv2
import time
import clip
import torch
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def delete_result():
    """clear old result in `.temp`"""
    results = sorted(os.listdir(CACHE_DIR))
    for res in results:
        if int(time.time()) - int(os.path.splitext(res)[0]) > 10000:
            os.remove(os.path.join(CACHE_DIR, res))


def download(img):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    while True:
        name = str(int(time.time()))
        tmp_name = os.path.join(CACHE_DIR, name + '.jpg')
        if not os.path.exists(tmp_name):
            break
        else:
            time.sleep(1)

    img.save(tmp_name, 'png')
    return tmp_name


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))



def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    gray_image = Image.new("RGB", image.size, (128, 128, 128))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    gray_image.paste(segmented_image, mask=transparency_mask_image)
    return gray_image


@torch.no_grad()
def image_text_match(cropped_objects, text_query):
    preprocessed_images = [preprocess(image).to(device) for image in cropped_objects]
    tokenized_text = clip.tokenize([text_query]).to(device)
    stacked_images = torch.stack(preprocessed_images)
    image_features = model.encode_image(stacked_images)
    text_features = model.encode_text(tokenized_text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100. * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)


def delete_result():
    """clear old result in `.temp`"""
    results = sorted(os.listdir(CACHE_DIR))
    for res in results:
        if int(time.time()) - int(os.path.splitext(res)[0]) > 10000:
            os.remove(os.path.join(CACHE_DIR, res))


def gradio_display(mask_generator, clip_model):
    import gradio as gr

    def clear_image_all():
        delete_result()
        return None, None, None, None

    def get_id_photo_output(image, text):
        """
        Get the special size and background photo.

        Args:
            img(numpy:ndarray): The image array.
            size(str): The size user specified.
            bg(str): The background color user specified.
            download_size(str): The size for image saving.

        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)
        
        cropped_objects = []
        image_pil = Image.fromarray(image)
        for mask in masks:
            bbox = [mask["bbox"][0], mask["bbox"][1], mask["bbox"][0] + mask["bbox"][2], mask["bbox"][1] + mask["bbox"][3]]
            cropped_objects.append(segment_image(image_pil, mask["segmentation"]).crop(bbox))

        scores = image_text_match(cropped_objects, str(text))
        text_matching_masks = []
        for idx, score in enumerate(scores):
            if score < 0.05:
                continue
            text_matching_mask = Image.fromarray(masks[idx]["segmentation"].astype('uint8') * 255)
            text_matching_masks.append(text_matching_mask)

        alpha_image = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
        alpha_color = (255, 0, 0, 180)

        draw = ImageDraw.Draw(alpha_image)
        for text_matching_mask in text_matching_masks:
            draw.bitmap((0, 0), text_matching_mask, fill=alpha_color)

        result_image = Image.alpha_composite(image_pil.convert('RGBA'), alpha_image)
        res_download = download(result_image)
        return result_image, result_image, res_download

    with gr.Blocks() as demo:
        gr.Markdown("""# 💎 Text to Segment Anything 💎 """)
        with gr.Tab("InputImage"):
            image_in = gr.Image(value=ID_PHOTO_IMAGE_DEMO, label="Input image")
            text = gr.inputs.Textbox(lines=1, placeholder=None, default="dog", label='💎 Input text prompt', optional=False)
            with gr.Row():
                image_clear_btn = gr.Button("Clear")
                image_submit_btn = gr.Button("Submit")

            with gr.Row():
                img_out1 = gr.Image(
                    label="Output image", interactive=False).style(height=300)
                img_out2 = gr.Image(
                    label="Output image with mask",
                    interactive=False).style(height=300)
            downloaded_img = gr.File(label='Image download').style(height=50)

        image_clear_btn.click(
            fn=clear_image_all,
            inputs=None,
            outputs=[image_in, text, img_out1, img_out2, downloaded_img])

        image_submit_btn.click(
            fn=get_id_photo_output,
            inputs=[image_in, text],
            outputs=[img_out1, img_out2, downloaded_img])

        gr.Markdown(
            """<font color=Gray>Tips: You can input the text prompt and try segment the default image OR upload any images you want to segment by click on the clear button first.</font>"""
        )

        gr.Markdown(
            """<font color=Gray>Segment Anything. 
            Refer to the [SAM](https://github.com/facebookresearch/segment-anything) for code strucure and model architecture.
            """
        )

        gr.Button.style(1)

    demo.launch(server_name="0.0.0.0", server_port=8078, share=True)


ID_PHOTO_IMAGE_DEMO = "./notebooks/images/dog.jpg"
CACHE_DIR = ".temp"


if __name__ == "__main__":
    model = "vit_h"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_generator = SamAutomaticMaskGenerator(sam_model_registry[model](checkpoint="../model/sam_vit_h_4b8939.pth").to(device))
    model, preprocess = clip.load("ViT-B/32", device=device)
    gradio_display(mask_generator, model)