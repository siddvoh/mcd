# Mutual Colormap Dependence

Code and data for *Mutual Colormap Dependence: How Recoloring Biomedical Images Changes Vision-Language Model Answers* by Siddharth Vohra, Runmin Jiang and Min Xu, accepted to the FABRIC workshop at WACV 2027. The paper and a plain-language walkthrough are at https://siddvoh.com/mutual-colormap-dependence/.

Four vision-language models answered 314 questions about 280 biomedical images, each shown in up to six colormaps. This repository holds all 20,880 answers and the code that turns them into every table and result in the paper. Rebuilding the results calls no model.

## Rebuilding the results

With Python 3.11 or newer:

```
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python analysis.py
```

This takes about five minutes on one core and writes `results/values.json` and the LaTeX tables in `results/tables`. The copies in this repository are its output.

## Data

`data/plan.json` describes the experiment: the four models with their OpenRouter identifiers and the provider that served each one, the eight cells that every model answered, the sampling settings, the seed that sets the order of the calls, and the exact prompts.

`data/items` holds the three panels. `slake50.csv` has 50 SLAKE questions on 16 images, `chest64.csv` has 64 ChestX-ray14 radiographs with one yes/no question each, and `bbbc200.csv` has 200 BBBC039 fluorescence images with their reference nucleus counts.

`data/answers.csv` has one row per call. `cell` names the condition, `colormap` and `repeat` say which rendering and which repeat the call was, and `answer`, `abstain` and `confidence` are the fields of the JSON object the model returned. For BBBC039, `answer` is the nucleus count. `reasoning_tokens` is the number of reasoning tokens the model reported, and 0 means none were reported.

The cells are named after their panel. The `raw` cells ask once under each of the six colormaps, the `same_gray` cells ask six times about the gray rendering, and the `reminder` cells repeat the six-map cell with one more sentence telling the model to ignore the colormap. `chest64_multirepeat` asks three times under each colormap, and its repeat 0 is the one-answer cell for ChestX-ray14.

`data/raw/chestxray14/images` holds the 64 radiographs as PNG files, read from `BahaaEldin0/NIH-Chest-Xray-14`, a Hugging Face mirror of the NIH release. `nih_image_index` in `chest64.csv` gives each radiograph's file name in the NIH release, so its answer can be checked against NIH's label file. A positive case carries the finding's label, a negative case lacks it but may carry other findings, and the opacity questions use the Effusion label.

## Code

`analysis.py` computes every result from `data/answers.csv` and the item files. `lut_validation.py` measures how the six colormap lookup tables behave after 8-bit quantization. `run.py` collects answers: it renders an image in one colormap, sends it with the prompts, keeps the first of up to ten replies that comes from the requested model, ends normally and is a complete JSON answer of the required form, stops if none does, and writes one row per call in the format of `data/answers.csv`.

## Running the experiment again

`run.py` calls paid models through OpenRouter, so it needs an API key, and it needs the source images. The radiographs are already here. The BBBC039 and SLAKE images come from their public releases:

```
curl -LO https://data.broadinstitute.org/bbbc/BBBC039/images.zip
unzip -q images.zip -x '__MACOSX/*' -d data/raw/microscopy
curl -LO https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/imgs.zip
unzip -q imgs.zip 'imgs/*/source.jpg' -d slake
mkdir -p data/raw/slake/images
for d in slake/imgs/xmlab*; do cp "$d/source.jpg" "data/raw/slake/images/${d##*xmlab}.jpg"; done
rm -r images.zip imgs.zip slake
```

Then, for one model and one cell:

```
export OPENROUTER_API_KEY=...
python run.py --model gpt_5_4_or --cell slake50_raw
```

The answers go to `runs/gpt_5_4_or/slake50_raw.csv`. Model names are the keys of `models` in `data/plan.json`, and cell names are the `name` fields of `cells`. The models sample at temperature 1.0, so a new run follows the same procedure but gives new answers.

## How the images were prepared

Each source image is read as one scalar channel. Color images are averaged over RGB, and 16-bit images are converted with Pillow's `convert("L")`, which clips values above 255. The BBBC039 images are 16-bit, and `pixels_above_255` in `bbbc200.csv` counts the clipped pixels of each one. The field is downscaled with Lanczos resampling only when its long edge exceeds 1024 pixels, then min-max normalized, passed through the matplotlib colormap and written as an 8-bit PNG by truncation. The BBBC039 reference count is the number of nuclei in the image's mask, counted as the 8-connected regions of equal nonzero value in its first channel, which keeps touching nuclei apart.

## Citation

```
@inproceedings{vohra2027colormap,
  title     = {Mutual Colormap Dependence: How Recoloring Biomedical Images
               Changes Vision-Language Model Answers},
  author    = {Vohra, Siddharth and Jiang, Runmin and Xu, Min},
  booktitle = {Workshop on Foundation AI for Biomedical Reasoning, Imaging, and
               Cognition (FABRIC) at WACV 2027},
  year      = {2027},
  address   = {Orlando, Florida, USA},
  note      = {To appear},
  url       = {https://siddvoh.com/mutual-colormap-dependence/}
}
```

## License

The code is under the MIT license in `LICENSE`. The dataset material keeps its own terms, which are listed in `NOTICE.txt`.
