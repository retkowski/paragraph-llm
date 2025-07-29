# Paragraph Insertion with LLMs

This repository contains the code used in the paper "*Paragraph Segmentation Revisited: Towards a Standard Task for Structuring Speech.*" It provides reproducible scripts that insert paragraph breaks into sentence-level transcripts while preserving the original text verbatim. 

## What's here

- `infer.py` runs paragraph insertion with provided sentence segmentation.

- `infer+sentenceseg.py` runs paragraph insertion with sentence tokenization through NLTK; with optional optional section-wise processing.

## Dependencies

Dependencies are provided in the `requirements.txt` and can be installed like this:

```bash
pip install -r requirements.py
```

## Configuration

In the script you will find this section to configure used model, input files and certain modes, edit as needed:

```py
model_id = "meta-llama/Llama-3.1-70B-Instruct"
torch_dtype = torch.float16
device_map = "auto"

# Data
project_folder = "./"
partitions = ('train',)
file_prefix = "tedpara."            # or "ytsegpara."
output_file_prefix = "tedpara-paragraphs."

# Behavior
DEBUG_MODE = False                  # True prints token tables; hits breakpoint() after each sentence
SECTION_BASED_SEGMENTATION = False  # only in infer+sentenceseg.py
```

Place `pandas` data files at repo root (or adjust `project_folder` in the script).

Expected columns per JSON row

- `text: List[str]`: sentence list of the document
- `targets`: optional; used by `infer+sentenceseg.py` only when `SECTION_BASED_SEGMENTATION=True`

## Running

After installation and configuration, running the script is as simple as:

```bash
python infer.py
```

and

```bash
python "infer+sentenceseg.py"
```

## Output

The script adds the following columns to the dataframe:
- `segmented_sections: List[List[List[str]]]`: List of `[section][paragraph][sentences]`
- `paragraph_targets: str`: Binary encoding of the paragraph boundaries

## License

This code and documentation are licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

You are free to share and adapt the material for any purpose, even commercially, as long as proper attribution is given.