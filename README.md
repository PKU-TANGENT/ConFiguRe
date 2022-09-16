# ConFiguRe: Exploring Discourse-level Chinese Figures of Speech

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
                "fos": "夸张" (Hyperbole),
                "begin": 44,
                "end": 65
            },
            {
                "figurativeUnit": "在这时候会飞出来觅食振翎，谈天说地般吱吱地叫个不休。",
                "fos": "比拟" (Personification),
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
  <img src="https://github.com/PKU-TANGENT/ConFiguRe/blob/main/figures/corpusInfo.jpg" width="450" alt="statistics">
</p>

## Model 

### Requirement

Run the following script to install the remaining dependencies,

```
pip install -r requirements.txt
```

### Experimental Results

## Citation

If you use ConFiguRe in your work, please cite our paper:

## Connection

If you have any questions, feel free to contact

- [dwzhu@pku.edu.cn;](dwzhu@pku.edu.cn)

- [lisujian@pku.edu.cn;](lisujian@pku.edu.cn)