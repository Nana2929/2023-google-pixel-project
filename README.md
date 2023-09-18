## Pixel ABSA TestSet (PATS)
### Description
- `PATS` is a complete ABSA dataset; each annotated entry is in the format of `(entity, aspect, sentiment, entity category, aspect category, sentiment polarity)`. The first 3 elements are extracted substrings of the original sentence. Note that when `aspect` is implicit, we do not label it, and hence that entry is reduced to
`(entity, sentiment, entity category, sentiment polarity)`.
- Raw data
    - The raw data is the data crawled by spiders in `crawler` branch.
    - Crawled times fall at 2022-09 to 2023-07.
- Annotation
    - The dataset is human-labeled by 3 people, following the [annotation guideline](https://docs.google.com/document/d/19w7FkId7zPuDumzs3LKfA632CLf0yIollKJHOhIDuMU/edit?usp=sharing).

### Statistics

| Forum | Total data size| Data with more than 1 entry | Total entries|
|----------|----------|----------|----------|
|  XDA |  63 |  13  |  78  |
|  Android Central  |  45 |  19  |  93  |
|  Ars Technica  |  54 | 7 | 22 |
|  Reddit | 43 |  8 |  49 |

### Overview
<img src="images/data_sample.png" width="400">

### Folder Structure
```bash
git switch data
```
- `processed` folder contains the label-studio exported labeled data, but processed.
- `src` folder contains the source code to convert the label-studio exported labeled data to the format that can be used by the below models.
- `scripts` folder contains the bash script to execute the `src`.
### Usage
```bash
python3 pats/src/convert.py      # for other prompts
python3 pats/src/alsc_convert.py # for ALSC prompt
```
- If you would like to change arguments, pass in the arguments as `python3 pats/src/convert.py --data_dir val1 --output_dir val2 ...`.
```python3
def main(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    inf_file_prefix: str = INF_FILE_PREFIX,
    eval_file_prefix: str = EVAL_FILE_PREFIX,
    prompt_format: str = "AS",
):
```
- For the available prompt formats, see [pats/src/constants.py: PROMPT_SWITCH](pats/src/constants.py).

