# Licences and attribution

**Code:** MIT (`LICENSE`). **Data:** not included in this repository; each corpus keeps its own licence, listed below. **Model weights:** see the last section. Literature the methods draw on is cited in `docs/voice-analyser-mk2-plan.md` §16 and in the decision log (`docs/decisions.md`).

This project is for personal and research use. Anyone wanting a commercial version can retrain with the code (MIT) on commercially licensed data; the training pools a model saw are recorded in its run directory (`config.json`).

## Corpora

| Corpus | Role | Licence | Cite |
|---|---|---|---|
| VocalSet v1.1 | training (resynthesis) | CC BY 4.0 | Wilkins, Seetharaman, Wahl & Pardo (2018). VocalSet: a singing voice dataset. ISMIR. |
| M4Singer | training (resynthesis), planned | CC BY-NC-SA 4.0 (`dataset_license.md` in the M4Singer repository) | Zhang et al. (2022). M4Singer: a multi-style, multi-singer and musical score provided Mandarin singing corpus. NeurIPS Datasets and Benchmarks. |
| Common Voice 27.0, Korean | training, planned | CC0; downloading it means agreeing not to try to identify its speakers | Ardila et al. (2020). Common Voice: a massively-multilingual speech corpus. LREC. |
| Zeroth-Korean | training, planned | CC BY 4.0 (OpenSLR 40) | Lucas Jo and Wonkyum Lee, Zeroth-Korean, OpenSLR SLR40, https://www.openslr.org/40/ |
| LibriTTS-R (train-clean-100, train-clean-360, dev-clean) | training (distillation audio, D-034); dev-clean held out for fidelity | CC BY 4.0 (OpenSLR 141) | Koizumi et al. (2023). LibriTTS-R: a restored multi-speaker text-to-speech corpus. Interspeech. Derived from LibriTTS (Zen et al. 2019) and LibriSpeech (Panayotov et al. 2015). |
| VCTK 0.92 | training, planned | CC BY 4.0 | Yamagishi, Veaux & MacDonald (2019). CSTR VCTK Corpus: English multi-speaker corpus for CSTR voice cloning toolkit (version 0.92). University of Edinburgh. doi:10.7488/ds/2645 |
| speechocean762 | training (distillation audio, D-034); half the speakers children | CC BY 4.0 (OpenSLR 101) | Zhang et al. (2021). speechocean762: an open-source non-native English speech corpus for pronunciation assessment. Interspeech. |
| Samrómur Children 21.09 | training (distillation audio, D-034); children aged 4–17, speakers never identified | CC BY 4.0 (OpenSLR 117) | Hernández Mena et al. (2022). Samrómur Children: an Icelandic speech corpus. LREC. Language and Voice Lab, Reykjavik University, with Almannarómur. |
| Multilingual LibriSpeech (Italian, Portuguese, Polish; Opus) | training (distillation audio, D-034) | CC BY 4.0 (OpenSLR 94) | Pratap et al. (2020). MLS: a large-scale multilingual dataset for speech research. Interspeech. |
| Hillenbrand et al. (1995) | evaluation only (gold) | free for research, distributed by J. Hillenbrand | Hillenbrand, Getty, Clark & Wheeler (1995). Acoustic characteristics of American English vowels. JASA 97(5). |
| PTDB-TUG | evaluation only (gold) | ODbL 1.0 (to verify) | Pirker, Wohlmayr, Petrik & Pernkopf (2011). A pitch tracking corpus with evaluation on multipitch tracking scenario. Interspeech. |
| MIR-1K | evaluation only (gold) | research use; no explicit licence found | Hsu & Jang (2010). On the improvement of singing voice separation for monaural recordings using the MIR-1K dataset. IEEE TASLP. |
| MDB-stem-synth | evaluation only (gold) | CC BY-NC 4.0 | Salamon, Bittner, Bonada, Bosch, Gómez & Bello (2017). An analysis/synthesis framework for automatic f0 annotation of multitrack datasets. ISMIR. |
| CMU ARCTIC (bdl, jmk, slt, with EGG) | evaluation only (closed-phase analysis, D-019) | CMU ARCTIC licence: any use, copyright notice kept (© Carnegie Mellon University 2003) | Kominek & Black (2003). CMU ARCTIC databases for speech synthesis. CMU-LTI-03-177. |
| Birkholz et al. (2019) supplementary audio (physical and simulated vowel resonators) | evaluation only (source-filter coupling, D-023); not redistributed | none stated (all rights reserved by default) | Birkholz, Gabriel, Kürbis & Echternach (2019). How the peak glottal area affects linear predictive coding-based formant estimates of vowels. JASA 146(1), 223–232. Audio from vocaltractlab.de. |
| OPENGLOT (repositories II and III: physically modelled vowels; vowels through 3D-printed vocal tract replicas) | evaluation only (source-tract coupling, D-024); not redistributed | free for research use with citation (project page) | Alku, Murtola, Malinen, Kuortti, Story, Airaksinen, Salmi, Vilkman & Geneid (2019). OPENGLOT – An open environment for the evaluation of glottal inverse filtering. Speech Communication 107, 38–47. https://research.spa.aalto.fi/projects/openglot/ |
| Saarbrücken Voice Database (healthy speakers and selected voice-disorder groups, with EGG) | evaluation only (voice quality, D-024) | CC BY 4.0 | Pützer & Barry, Institute of Phonetics, Saarland University; hosted by Essen University Hospital. Zenodo, doi:10.5281/zenodo.16874898. |
| MOCHA-TIMIT (fsew0, msak0, maps0; with laryngograph) | evaluation only (closed-phase references); not redistributed | research and educational use, no commercial use without permission; licence file kept with the data | Wrench (1999). MOCHA-TIMIT articulatory database. Queen Margaret University College, Edinburgh. https://data.cstr.ed.ac.uk/mocha/ |
| USC 75-Speaker Speech MRI Database (`Subjects.xlsx` so far: id, sex, age, height only) | evaluation only (anatomical VTL, D-035), planned | CC BY 4.0 (figshare) | Lim, Toutios, et al. (2021). A multispeaker dataset of raw and reconstructed speech production real-time MRI video and 3D volumetric images. Scientific Data 8, 187. doi:10.6084/m9.figshare.13725546 |
| 75-Speaker Annot-16 (hand-drawn midsagittal contours, 160 frames of 16 speakers) | evaluation only (MRI midline VTL, D-035); not redistributed, derived per-speaker values kept outside the repo | Zenodo says CC BY 4.0, but the README's agreement says no redistribution, individual research use only; the stricter reading is followed | Shi, Zhang, Lu, Ma, Feng, Toutios, Hsu, Goldstein & Narayanan (2025). 75-Speaker Annot-16: a benchmark dataset for speech articulatory rt-MRI annotation with articulator contours and phonetic alignment. Interspeech, 2175–2179. Zenodo record 18931763. |

Evaluation-only corpora are never trained on (plan §6.4, enforced by `va leakage`), so they don't bear on the weights' licence. Synthetic training data is generated by this code.

## Model weights

Whether trained weights are an adaptation of their training data is legally unsettled. This project takes the conservative reading: **weights carry the most restrictive licence among the corpora they were trained on**, plus attribution to all of them.
- Synthetic data, VocalSet, Zeroth-Korean, Common Voice (and other CC0/CC BY pools): **CC BY 4.0**, attributing those corpora.
- Anything that includes M4Singer: **CC BY-NC-SA 4.0**.
- Distilled students (D-034) are also trained on the outputs of NVIDIA's `parakeet-tdt-0.6b-v3` (CC BY 4.0; weights fetched directly, not redistributed): **CC BY 4.0**, attributing NVIDIA (Parakeet TDT 0.6B v3, NeMo) as well as the audio corpora.

If permissively licensed weights are wanted later, train a build without the NC-SA pools; the code makes that a configuration change.

*This is a good-faith reading of the licences, not legal advice.*
