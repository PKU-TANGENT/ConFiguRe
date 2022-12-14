# ConFiguRe
 
This repository presents the dataset and baseline implementation for the COLING 2022 long paper (oral): "ConFiguRe: Exploring Discourse-level Chinese Figures of Speech".

## Overview

ConFiguRe is a Chinese corpus for Contextualized Figure Recognition concerning 12 commonly used rhetoric figures. It involves both extracting a figurative unit from the discourse-level context and classifying this unit into the right figure type. On the basis of ConFiguRe, three tasks are devised as benchmarks, i.e. *figure extraction*, *figure type classification* and *figure recognition*.

## ConFiguRe Dataset

ConFiguRe dataset situates within the `data` folder of this repository. Train, valid and test set can be find in `data/train.json`, `data/valid.json`, `data/test.json`, respectively.  The 12 figure types in ConFiguRe are *Metaphor/Simile*, *Personification*, *Metonymy*, *Hyperbole*, *Irony*, *Synaesthesia*, *Rhetorical question*, *Parallelism*, *Duality*, *Repetition*, *Antithesis*, and *Quote*. For definition and example of each figure type, please refer to the original paper.

### Data Format

Each instance of ConFiguRe comprises two parts: *fragment* and *units*.  The former is piece of Chinese literary works containing figures of speech (fos). The latter is a collection of all the figurative units in the fragment.

A sample data point in the json file is as follows: 

```json
{
    "北平的四季-郁达夫_片段4": {
        "fragment": "到了下雪的时候哩，景象当然又要一变。早晨从厚棉被里张开眼来，一室的清光会使你的眼睛眩晕。在阳光照耀之下，雪也一粒一粒地放起光来了，蛰伏得很久的小鸟，在这时候会飞出来觅食振翎，谈天说地般吱吱地叫个不休。数日来的灰暗天空，愁云一扫，忽然变得澄清见底，翳障全无；于是，年轻的北方住民，就可以营屋外的生活了——溜冰，做雪人，赶冰车雪车……就在这一种日子里最有劲儿。我曾于这一种大雪时晴的傍晚，和几位朋友跨上跛驴，出西直门上骆驼庄去过过一夜。北平郊外的一片大雪地，无数枯树林，以及西山隐隐现现的不少白峰头，和时时吹来的几阵雪样的西北风，所给予人的印象实在是深刻、伟大，神秘到了不可以言语来形容。直到了十余年后的现在，我一想起当时的情景，还会打一个寒战而吐一口清气，如同在钓鱼台溪旁立着的一瞬间一样。",
        "units": [
            {
                "figurativeUnit": "在阳光照耀之下，雪也一粒一粒地放起光来了，",
                "fos": "夸张 (Hyperbole)",
                "begin": 44,
                "end": 65
            },
            {
                "figurativeUnit": "在这时候会飞出来觅食振翎，谈天说地般吱吱地叫个不休。",
                "fos": "比拟 (Personification)",
                "begin": 74,
                "end": 100
            }
        ]
    },
}
```

### Corpus Statistics

ConFiguRe includes 4,192 fragments and 9,010 figurative units. Train, valid and test set is split according to the proportion of 7:1:2. Detailed information for each figure type is demonstrated below:

<p align="center">
  <img src="./graphs/corpusInfo.png" width="450" alt="statistics">
</p>

## Environment
### Overall Information
- `transformers` version: 4.18.0
- Platform: Linux-5.4.0-124-generic-x86_64-with-glibc2.17
- Python version: 3.8.13
- Huggingface_hub version: 0.5.1
- PyTorch version (GPU?): 1.11.0+cu113 (True)

### Reproduction
```bash
conda env create -n configure python=3.8.13 -y
pip install -r requirements.txt
```
## Code Layout
We provide a general overview of our code repo. For detailed annotation, please refer to the comments in each file.
```bash
/
├── configs/ # yaml style configs
│   ├── accelerate_config.yaml # sample config for huggingface accelerate module
│   ├── default.yaml # default training config
│   ├── hydra/
│   │   └── job_logging/
│   │       └── custom.yaml # handles auto logging
│   └── model_args/ # task specific config
│       ├── Classification.yaml
│       ├── ClassificationContext.yaml
│       ├── CRF.yaml
│       ├── End2end.yaml
│       └── Extraction.yaml
├── delimit_clause.py # logic for delimiting clauses, which would serve as basis for `figurative unit`
├── dataset/ # handles dataset loading
│   ├── __init__.py
│   ├── DatasetForClassification.py
│   ├── DatasetForClassificationContext.py
│   ├── DatasetForCRF.py
│   ├── DatasetForExtraction.py
│   └── DatasetForRecognition.py
├── main.py # template for training
├── metrics/ # handles metric calculation
│   ├── __init__.py
│   ├── MetricForClassification.py
│   ├── MetricForExtraction.py
│   ├── MetricForRecognition.py
│   └── MetricForRecognitionCRF.py
├── model/ # implementation for models, including forward logic
│   ├── __init__.py
│   ├── BertForFigClassification.py
│   ├── BertForFigClassificationContext.py
│   ├── BertForFigExtraction.py
│   ├── BertForFigExtractionContrast.py
│   ├── BertForFigRecognition.py
│   └── BertForFigRecognitionCRF.py
├── scripts/ # helpful scripts
│   ├── debug_hydra.sh
│   ├── eval.sh
│   └── run.sh
├── task/ # task specific logic, called in `main.py`
│   ├── __init__.py
│   ├── Classification.py
│   ├── ClassificationContext.py
│   ├── CRF.py
│   ├── End2end.py
│   ├── Extraction.py
│   ├── ExtractionContrast.py
│   └── ExtractionCRF.py
└── train/ # Trainer-like module, handles training + eval steps
    ├── __init__.py
    ├── TrainClassifier.py
    ├── TrainExtraction.py
    ├── TrainRecognition.py
    └── TrainRecognitionCRF.py
```
### Hydra Config
We leverage the [hydra module](https://hydra.cc/) to store hyperparameters with respect to each model, to enable auto-logging and to modularize our repo. It is helpful to have a basic idea of the hydra configuration style. 
### Scripts
Useful scripts are under the `code/scripts` folder. 

Navigate to `code/` folder, and run the scripts, eg.
```bash
# cd code
bash scripts/run.sh
```
## Experimental Results
<p align="center">
  <img src="./graphs/mainResults.png" width="700" alt="statistics">
</p>

## Citation

If you use ConFiguRe in your work, please cite our paper:

```bibtex
@misc{https://doi.org/10.48550/arxiv.2209.07678,
  doi = {10.48550/ARXIV.2209.07678},
  url = {https://arxiv.org/abs/2209.07678},
  author = {Zhu, Dawei and Zhan, Qiusi and Zhou, Zhejian and Song, Yifan and Zhang, Jiebin and Li, Sujian},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {ConFiguRe: Exploring Discourse-level Chinese Figures of Speech},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## Contact

If you have any questions, feel free to open an issue, or contact

- [dwzhu@pku.edu.cn;](mailto:dwzhu@pku.edu.cn)

- [lisujian@pku.edu.cn;](mailto:lisujian@pku.edu.cn)

For implementation details

- [zhouzhejian@pku.edu.cn;](mailto:zhouzhejian@pku.edu.cn)
