# 2023 Google Pixel User Sentiment Monitoring
<img src="https://img.shields.io/badge/google_pixel-_2023v1-blue" alt="google-pixel-2023">

### Links
- [2023 Progress Reports @ Pixel User Sentiment Monitoring Sync](https://drive.google.com/drive/folders/1FHPTiFqAyFaGuS5PuSO9JkydBFkhk2tB)
### 1. Crawler
- 🗓 reported date:
    - [2022/09/26](https://docs.google.com/presentation/d/1zjBteSB8EfMOmQVfnQCxsttVc5gi5PItEjCUVVchewM/edit?usp=drive_link)
    - [2022/10/12](https://docs.google.com/presentation/d/10kmpzegqsEFibcmgR7r9-X0jcLCHpRUCY4hD9XQJ1Sw/edit?usp=drive_link)
- 🌲 branch: `crawler`
- 🗂️ The crawler to crawl the data from the Google Pixel forum.
- 🔩 toolkits:
    - framework: [scrapy](https://scrapy.org/doc/).
### 2. Pixel ABSA TestSet (PATS)
- 🗓 reported date:
    - [2023/08/04](https://docs.google.com/presentation/d/18XPeXfnfvN8gz9ybVSwB9pV49PzIi6h7wyGWVdzUZ5E/edit?usp=drive_link)
    - [2023/08/17](https://docs.google.com/presentation/d/1-HrWtk0kpsKoZyJUs2rz0FMkUcVUGy5YoOdlG2y75XU/edit#slide=id.g15935f97167_0_408)
- 🌲 branch: `pats`
- 🗂️ Labeled data of the crawled data.
- 🔩 toolkits:
    - labeling platform: [Label Studio](https://labelstud.io/)
- The scripts inside convert the labeled data from label-studio format to the format that can be used by (1) bert-based models: [RobertaABSA model](https://github.com/ROGERDJQ/RoBERTaABSA) and (2) text-to-text models: [LLaMA](https://github.com/facebookresearch/llama).
### 3. BERT-Series: A Simple ABSA Baseline
- 🗓 reported date:
    - [2023/02/17](https://docs.google.com/presentation/d/13jtvRyBVg8q7Oqb01dBL6qXspmugSf1w2WfPXPK85DU/edit?usp=drive_link)
- 🌲 branch: `bert_baseline`
- 🗂️ A training and testing script for running BERT-based ATE+ALSC pipeline for simple aspect-based sentiment analysis.
- 🔩 toolkits:
    - framework: [PyTorch Lightning](https://lightning.ai/) + [miracleyoo/pytorch-lightning-template](https://github.com/miracleyoo/pytorch-lightning-template/tree/master/classification)
### 4. LLM-Series: ABSA with ChatGPT
- 🗓 reported date:
    - [2023/04/14](https://docs.google.com/presentation/d/1yN4j-pvaQdlNlZ-HM5lI8pSOTjMp-CD1gBNcFVga1RM/edit?usp=drive_link)
- 🌲 branch: N/A
- 🔩 toolkits:
    - data analysis tool: [tableau](https://www.tableau.com/zh-tw)
- 🗂️ content:
    - visualization: [Pixel Sentiment Monitoring gpt baseline @ Tableau Public](https://public.tableau.com/views/PixelSentimentMonitoringgptbaseline/SentimentDashboard?:language=zh-TW&:display_count=n&:origin=viz_share_link)
    - report: [2023/04/14 Progress Report p.6-11](https://docs.google.com/presentation/d/1yN4j-pvaQdlNlZ-HM5lI8pSOTjMp-CD1gBNcFVga1RM/edit#slide=id.g1f5dd1e8b94_1_12)
### 5. LLM-Series: Senti-LLaMA Experiments [to be updated]
- 🗓 reported date:
    - [2023/07/07](https://docs.google.com/presentation/d/101qrt2zNZkEst-tIk19sNlvU9B-xh317Tjpej4F1dEs/edit?usp=drive_link) ~ [2023/09/15](https://docs.google.com/presentation/d/1DFgG1iD23QbhNpcA0N9oE_6pG_LnkiCbgCbplCHbvjQ/edit?usp=drive_link)
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

### 6. Dialogue ABSA
- 🗓 reported date:
    - [2023/09/15](https://docs.google.com/presentation/d/1DFgG1iD23QbhNpcA0N9oE_6pG_LnkiCbgCbplCHbvjQ/edit)
    - [2023/06/09](https://docs.google.com/presentation/d/1ilaPOG7gICbYqB9M_78gwkdSb8vg79XEaGnLi-4V56Y/edit?usp=drive_link)
- 🌲 repo: [Nana2929/dialogue-absa](https://github.com/Nana2929/dialogue-absa/tree/main)
- 🗂️ content:
    - We test the dialogue absa dataset [`DiaASQ`](https://github.com/unikcc/DiaASQ) with 3 models. See the repo for more details.
        - analysis: [0609report](https://docs.google.com/presentation/d/1ilaPOG7gICbYqB9M_78gwkdSb8vg79XEaGnLi-4V56Y/edit#slide=id.g226f51f7eda_0_2), [0915 report](https://docs.google.com/presentation/d/1DFgG1iD23QbhNpcA0N9oE_6pG_LnkiCbgCbplCHbvjQ/edit#slide=id.g27f0814905a_0_0)
        - [statistics](https://docs.google.com/spreadsheets/d/1lEKXvQBUi7Ya0MGoMtZYQ3vFJrGHhl9M0ZoVLa-h4RQ/edit?usp=sharing)
    - 🌲 branch: `data`: conversion scripts for generative model (in our case, LLaMA-2) training.
    - 🌲 branch: `gpt`: api calling scripts for ChatGPT.
    - 🌲 branch: `llama`: metrics scripts for generative model (in our case, all) inference.
    - 🌲 branch: `train`: training scripts for T5.
- 🔩 toolkits:
    - framework: [PyTorch Lightning](https://lightning.ai/) + [miracleyoo/pytorch-lightning-template](https://github.com/miracleyoo/pytorch-lightning-template/tree/master/classification)
