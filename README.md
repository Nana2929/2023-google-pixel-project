# 2023 Google Pixel User Sentiment Monitoring

- [Experiment statistics]()
- [2023 progress reports @ Pixel User Sentiment Monitoring Sync](https://drive.google.com/drive/folders/1FHPTiFqAyFaGuS5PuSO9JkydBFkhk2tB)

## Content

1. Crawler
    - The crawler to crawl the data from the Google Pixel forum.
    - Crawlers are based on [scrapy](https://scrapy.org/doc/).
2. Pixel Dataset
    - Label data of the crawled data
    - The scripts to convert the labeled data from label-studio format to the format that can be used by the below models
       - RobertaABSA
       - Senti-LLaMA
3. Senti-LLaMA Experiments
    - Training script and hyperparam configs.
    - Sentiment-LLaMA weights (?)
    - Sentiment-LLaMA format converted dataset
4. A Simple Baseline


## 1. Crawler
- Python Scrapy crawlers to crawl the 4 forums for Pixel Dataset.
- Execute the scrapy commands following [official documentation](https://scrapy.org/doc/).
    - At the top level of each scrapy project, execute the following. eg.
    ```shell
    // at XDA
    scrapy crawl XDA
    ```

## 2. Pixel Dataset
<!-- nanaeilish projects/sa-instruction-tuning -->
### Description
- The dataset is human-labeled by 3 people, following the [annotation guideline](https://docs.google.com/document/d/19w7FkId7zPuDumzs3LKfA632CLf0yIollKJHOhIDuMU/edit?usp=sharing).
- The dataset is a complete ABSA dataset; each annotated entry is in the format of `(entity, aspect, sentiment, entity category, aspect category, sentiment polarity)`. The first 3 elements are extracted substrings of the original sentence. Note that when `aspect` is implicit, we do not label it, and hence that entry is reduced to
`(entity, sentiment, entity category, sentiment polarity)`.

| Forum | Total data size| Data with more than 1 entry | Total entries|
|----------|----------|----------|----------|
|  XDA |  63 |  13  |  78  |
|  Android Central  |  45 |  19  |  93  |
|  Ars Technica  |  54 | 7 | 22 |
|  Reddit | 43 |  8 |  49 |

### Folder Structure
```
git switch data
```
- `processed` folder contains the label-studio exported labeled data, but processed.
- `src` folder contains the source code to convert the label-studio exported labeled data to the format that can be used by the below models.
- `scripts` folder contains the bash script to execute the `src`.


## 3. Senti-LLaMA Experiments