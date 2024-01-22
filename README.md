# Music-Data-Leakage

The Large Scale Audio Foundation model plays a crucial role in the field of Music Information Retrieval. To enhance its generalization capabilities, the model undergoes pretraining using extensive multi-source datasets. However, there is currently a lack of thorough investigation into the overlap between the test dataset used to evaluate the performance of pretrained models and the pretraining dataset. This gap in examining potential data leakage or contamination poses a significant challenge, as it can lead to a misguided evaluation of the pretraining model's efficacy, especially in terms of its ability to generalize to out-of-distribution datasets.

[![image](https://i.imgur.com/nz1yBU4.png)]()

This repository employs two methods to check for data leakage.

### Metadata Feature

The first method involves measuring the similarity of {title} by {artist} text in data with metadata using Sentence BERT. Samples with similarity scores above a certain threshold are provisionally designated as potential data leakage samples. In this case, datasets such as MSD, MTT, FMA, and EmoMusic can be utilized.

```
python get_metadata_embedding.py
python match_by_metadata.py
```

### Audio FingerPrinting
The second method utilizes [audio fingerprinting](https://github.com/dpwe/audfprint) to measure the overlap between the pretrain dataset and query dataset even in the absence of metadata. We use the audiofp library to check for data leakage based on hash overlap ratios.

```
Update soon
```

The results are as follows and are gathered in the "overlap" folder.

| pretrain_dataset (train split) | downstream_dataset (test split) | overlap_sample (downstream) |
|------------------|--------------------|-----------------------------|
| Audioset         | Music_caps         | 23                          |
| FMA_Large        | EmoMusic           | 553                         |
| MSD              | MTT                | 1807                        |
| MSD              | GTZAN              | 159                         |
| Music4ALL        | MTT                | 31                          |
| Music4ALL        | GTZAN              | 94                          |