# GPT-4V with Emotion



## Goal

We evaluate the performance of GPT-4V in multimodal emotion understanding. To the best of our knowledge, this is the first work to quantitatively evaluate the performance of GPT-4V on emotional tasks. We hope that our work can establish a zero-shot benchmark for subsequent research and inspire future directions in affective computing.

Details can be found in our paper: [**GPT-4V with Emotion: A Zero-shot Benchmark for Multimodal Emotion Understanding**](https://arxiv.org/pdf/2312.04293.pdf)

```tex
@article{lian2023explainable,
  title={GPT-4V with Emotion: A Zero-shot Benchmark for Multimodal Emotion Understanding},
  author={Lian, Zheng and Sun, Licai and Sun, Haiyang and Chen, Kang and Wen, Zhuofan and Gu, Hao and Chen, Shun and Liu, Bin and Tao, Jianhua},
  journal={arXiv preprint arXiv:2312.04293},
  year={2023}
}
```



## Evaluation Tasks

![dataset-1](H:\desktop\Multimedia-Transformer\gp4v-emotion\image\dataset-1.png)

## Supervised Models vs. Zero-shot GPT-4V

![result1](H:\desktop\Multimedia-Transformer\gp4v-emotion\image\result1.png)

![result2](H:\desktop\Multimedia-Transformer\gp4v-emotion\image\result2.png)

![result3](H:\desktop\Multimedia-Transformer\gp4v-emotion\image\result3.png)

![result4](H:\desktop\Multimedia-Transformer\gp4v-emotion\image\result4.png)

![result5](H:\desktop\Multimedia-Transformer\gp4v-emotion\image\result5.png)

## Request for GPT-4V

1. config.py: add your OpenAI key into the **'candidate_keys' **

   Note: We support multiple keys. The model can automatically change the key when it meets the daily request limit.

2. main.py: change **'dataset'** and **'save_root'** into your own path
3. dataset preprocess

```
# facial emotion recognition
1. create a 'image' folder to store test samples

# visual sentiment analysis
1. create a 'evoke' folder to store test samples

# micro-expression recognition
1. create a 'micro' folder to store test samples

# dynamic facial emotion recognition
1. create a 'video' folder to store test samples

# multimodal emotion recognition
1. create a 'video' folder to store test videos, named as {filename}.avi or {filename}.mp4
2. create a 'text' folder to store test text, named as {filename}.npy
```

4. run **'python main.py'** for evaluation

   

We provide raw prediction results of GPT-4V in the **results** folder
