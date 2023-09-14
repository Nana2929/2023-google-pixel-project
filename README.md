# 2023 Google Pixel User Sentiment Monitoring

- [2023 Progress Reports @ Pixel User Sentiment Monitoring Sync](https://drive.google.com/drive/folders/1FHPTiFqAyFaGuS5PuSO9JkydBFkhk2tB)
### 1. Crawler
- 🌲 branch: `crawler`
- 🗂️ The crawler to crawl the data from the Google Pixel forum.
- 🔩 toolkits:
    - framework: [scrapy](https://scrapy.org/doc/).
### 2. Pixel ABSA TestSet (PATS)
- 🌲 branch: `pats`
- 🗂️ Labeled data of the crawled data.
- 🔩 toolkits:
    - labeling platform: [Label Studio](https://labelstud.io/)
- The scripts inside convert the labeled data from label-studio format to the format that can be used by (1) bert-based models: [RobertaABSA model](https://github.com/ROGERDJQ/RoBERTaABSA) and (2) text-to-text models: [LLaMA](https://github.com/facebookresearch/llama).
### 3. Senti-LLaMA Experiments [to be updated]
- 🌲 branch: `senti_llama`
- 🗂️ content:
    - training script and hyperparam configs for finetuning LLaMA 2 (-chat) on sentiment datasets
    - sentiment-LLaMA weights
    - sentiment-LLaMA input-format converted datasets
        - yelp, imdb, ... [to be updated]
- 🔩 toolkits:
    - training streamline: [OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
    - attention mechanism: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
    - efficiency tool: [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)
### 4. A Simple ABSA Baseline
- 🌲 branch: `bert_baseline`
- 🔩 toolkits:
    - framework: [PyTorch Lightning](https://lightning.ai/) + [miracleyoo/pytorch-lightning-template](https://github.com/miracleyoo/pytorch-lightning-template/tree/master/classification)
### 5. ChatGPT ABSA
- 🌲 branch: N/A
- 🔩 toolkits:
    - data analysis tool: [tableau](https://www.tableau.com/zh-tw)
- 🗂️ content:
    - visualization: [Pixel Sentiment Monitoring gpt baseline @ Tableau Public](https://public.tableau.com/views/PixelSentimentMonitoringgptbaseline/SentimentDashboard?:language=zh-TW&:display_count=n&:origin=viz_share_link)
    - report: [2023/04/14 Progress Report p.6-11](https://docs.google.com/presentation/d/1yN4j-pvaQdlNlZ-HM5lI8pSOTjMp-CD1gBNcFVga1RM/edit#slide=id.g1f5dd1e8b94_1_12)
