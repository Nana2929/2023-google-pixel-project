# 2023 Google Pixel User Sentiment Monitoring

- [2023 Progress Reports @ Pixel User Sentiment Monitoring Sync](https://drive.google.com/drive/folders/1FHPTiFqAyFaGuS5PuSO9JkydBFkhk2tB)

## Content
1. Crawler
    - branch: `crawler`
    - The crawler to crawl the data from the Google Pixel forum.
    - Crawlers are based on [scrapy](https://scrapy.org/doc/).
2. Pixel ABSA TestSet (PATS)
    - branch: `data`
    - Label data of the crawled data
    - The scripts to convert the labeled data from label-studio format to the format that can be used by the below models
       - RobertaABSA
       - Senti-LLaMA
3. Senti-LLaMA Experiments [to be updated]
    - branch:
    - Training script and hyperparam configs.
    - Sentiment-LLaMA weights (?)
    - Sentiment-LLaMA format converted dataset
4. A Simple ABSA Baseline
    - branch: `bert_baseline`
5. ChatGPT ABSA
    - branch: N/A
    - Visualization: [Pixel Sentiment Monitoring gpt baseline @ Tableau Public](https://public.tableau.com/views/PixelSentimentMonitoringgptbaseline/SentimentDashboard?:language=zh-TW&:display_count=n&:origin=viz_share_link)
    - Analysis: [2023/04/14 Progress Report p.6-11](https://docs.google.com/presentation/d/1yN4j-pvaQdlNlZ-HM5lI8pSOTjMp-CD1gBNcFVga1RM/edit#slide=id.g1f5dd1e8b94_1_12)
