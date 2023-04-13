# Text to Segment Anything

## Segment Anything with Text Prompts

**Segment Anything Model (SAM)**  can produce high quality object masks from different types of prompts including points, boxes, masks and text. Unfortunately, the text prompt SAM model is not released. Therefore, we used a combination of **SAM** and **CLIP** to calculate the similarity between the output masks and text prompt. In this way, you can use **text prompt** to segment anything.

<div align="center">
<img src="https://user-images.githubusercontent.com/18344247/231759722-84ddc7cf-f1a7-4156-94cc-24853baac937.jpg"  width="1000" />
</div>


## Run the demo

start a gradio service with the following scrip on local machine and you can try out our project with your own images.

```bash
python3 text_sam.py --checkpoint_path ../model/sam_vit_h_4b8939.pth model_type vit_h

```


## Reference

> Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick. [Segment Anything](https://ai.facebook.com/research/publications/segment-anything/).

> Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever Proceedings of the 38th International Conference on Machine Learning, PMLR 139:8748-8763, 2021. [CLIP](https://github.com/openai/CLIP)