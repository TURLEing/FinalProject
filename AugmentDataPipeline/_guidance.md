## 有关抑郁数据集的数据增强 pipeline 搭建指导

- 观察了一下，剔除所有长度小于1s的方案是可行的。

- 搭建一个自动训模型的 pipeline：

### 1. 数据预处理

  - 先从对应语音文件夹中找出符合长度要求的，copy 到文件夹（slicer_opt）

  - 调用 denoise.py，转到 asr.py 然后再看看 webui 是怎么一个训练流程，copy 一下环境参数；

  - 或者不用 asr.py，直接把数据格式统一成他那个样子，对应表格存储在 output/asr_opt/denoise_opt.list.

  - 最后执行【一键三连】，格式化数据：
    - 1-get-text.py
    - 2-get-hubert-wav32k.py
    - 3-get-semantic.py

### 2. 训练过程：sovits train & gpt train
`GPT_SoVITS/s2_train.py --config "/home/jovyan/GPT-SoVITS/TEMP/tmp_s2.json"`，参数: 12 epochs

`GPT_SoVITS/s1_train.py --config_file "/home/jovyan/GPT-SoVITS/TEMP/tmp_s1.yaml"`，参数：20 epochs

*PS: 训练时可以考虑仅勾选保存最近的，也可以看看效果。*

最终版的模型存放路径在 `GPT_weights_v2` 和 `SoVITS_weights_v2` 下。

### 3. 推理过程

推理已经搭好框架了，直接调用即可。

可以先用工具找到最合适的参考文本。

生成结束后，可以借助 asr 工具数据清洗，判断生成质量，进行筛选：可能遇到复读和参考泄露问题。

### 参考 1：

建议将如何代码统一放入到工具类中，包括参数控制（目前已经有部分是的），可以将与模型加载的动作独立出来：
slice
asr
一键三连中的三步
sovits train
gpt train

inference

这样在webui或者api中，只通过函数调用；这样封装另外一个cmd的入口就会比较简单