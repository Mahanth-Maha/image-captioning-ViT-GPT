

# Image Captioning 

## Literature Work:

> Snippets from https://arxiv.org/pdf/2004.14231 : Sec 2
### Single-Stage Attention Based Image Captioning
```
The availability of large-scale annotated datasets enabled the training of deep models for image captioning. Vinyals et al. [18] proposed the first deep model for image captioning. Their model uses a CNN pre-trained on ImageNet [16] to encode the image, then a LSTM [19] based language model is used to decode the image features into a sequence of words. Xu et al. [1] introduced an attention mechanism into image captioning during the generation of each word, based on the hidden state of their language model and the previous generated word. Their attention module generates a matrix to weight each receptive field in the encoded feature map, and then feed the weighted feature map and the previous generated word to the language model to generate the next word. Instead of only attending to the receptive field in the encoded feature map, Chen et al. [2] added a feature channel attention module, their channel attention module re-weight each feature channel during the generation of each word. Not all the words in the sentence have a correspondence in the image, so Lu et al. [20] proposed an adaptive attention approach, where their model has a visual sentinel which adaptively decides when and where to rely on the visual information.


The single-stage attention model is computational efficient, but lacks accurate positioning of informative regions in the original image.
```
### Two-Stages Attention Based Image Captioning
```
Two stage attention models consists of bottom-up attention and top-down attention, where bottom-up attention first uses object detection models to detect multiple informative regions in the image, then top-down attention attends to the most relevant detected regions when generating a word.

The performance of two-stage attention based image captioning models is improved a lot against single-stage attention based models. However, each detected region is isolated from others, lacking the relationship with other regions.
```
### Visual Scene Graph Based Image Captioning
```
Visual scene graph based image captioning models extend two-stage attention models by injecting a graph convolutional neural network to relate detected informative regions, and therefore refine their features before feeding into the decoder.


Introducing the graph neural network to relate informative regions yields a sizeable performance improvement for image captioning models, compared to two-stage attention models. However, it requires auxiliary models to detect and build the scene graph at first. Also those models usually have two parallel streams, one responsible for the semantic scene graph and another for spatial scene graph, which is computationally inefficient.
```

### Transformer Based Image Captioning
```
Transformer based image captioning models use the dot-product attention mechanism to relate informative regions implicitly.

In image captioning, AoANet [https://arxiv.org/abs/1908.06954] uses the original internal transformer layer architecture, with the addition of a gated linear layer [https://arxiv.org/pdf/1612.08083] on top of the multi-head attention. The object relation network [14] injects the relative spatial attention into the dot-product attention. Another interesting result described by Herdade et al. [14] is that the simple position encoding (as proposed in the original transformer) did not improve image captioning performance. The entangled transformer model [13] features a dual parallel transformer to encode and refine visual and semantic information in the image, which is fused through gated bilateral controller.

Compared to scene graph based image captioning models, transformer based models do not require auxiliary models to detect and build the scene graph at first, which is more computational efficient. 
```





## Papers 

1. `AoA` Net (2019): Attention on Attention for Image Captioning 
- https://arxiv.org/abs/1908.06954

2. `ImgT` (2020) : Image Captioning through Image Transformer
- https://arxiv.org/pdf/2004.14231 
- https://github.com/wtliao/ImageTransformer 

3. Clip(VIT)-GPT(2) (2022) : A Frustratingly Simple Approach for End-to-End Image Captioning
- 


# Topics 

### GLU 
- https://arxiv.org/pdf/2002.05202
- https://www.reddit.com/r/MachineLearning/comments/1b6ggpz/d_why_do_glus_gated_linear_units_work/
>  We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence.