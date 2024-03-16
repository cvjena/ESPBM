# ESPBM - Eye State Prototype Blink Matching

![Prototype Learning and Manual Definition](assets/prototypes.png)

This work presents a novel method for detecting eye blinking by establishing *eye state prototypes* to match blink patterns within the eye aspect ratio (EAR) time series.
In contrast to traditional methods, which mainly focus on the binary ON/OFF of blinks, our method takes care of critical diagnostic details such as blink speed, duration, and inter-eye synchronicity.

In an unsupervised manner, we learned prototypes from the existing blink patterns and established manually defined prototypes.
Our research shows that both *unsupervised learned* and *manually defined prototypes* can reliably detect blink intervals and have comparable results, which offers potential diagnostic tools for identifying muscular or neural disorders.

Under the "minimal working prototype" principle, we aim to establish the eye blink prototype with minimal work, enabling medical professionals without computer expertise to quickly create prototypes to match specific patterns.
This repository presents the source code of our approach and provides a demonstration of sample experiments.

## Getting Started

### Locally

1. Clone the repository

```bash
git clone <link>
```

2. `cd` to EBPM
3. Create a new conda environment and install the dependencies:

```bash
conda create -n ebpm python=3.10 -y
conda activate ebpm
# conda install cudatoolkit -y
pip install jupyter
pip install -e .
```

## Via Pip

```bash
Upcoming
```

## Usage

We provide some example time series and prototypes to demonstrate the usage of our method in the `sample_experiments` folder.
Our code is designed to be modular and flexible, so you can quickly adapt it to your data and prototypes.
We demonstrate the usage of our method in the following steps:

- Data loading
- Prototypes learning or manual definition
- Extraction
- Inter-eye synchronicity

## Citation

```bibtex
Upcoming
```

## License

Licensed under [MIT](License.txt).

## Acknowledgments

Our project is and was only possible with the help of many existing projects and their maintainers and contributors.
We would like to thank the following projects and their maintainers for their work:

- [Stumpy](https://github.com/TDAmeritrade/stumpy) for the unsupervised learning of prototypes and fast pattern-matching
- [JeFaPaTo](https://github.com/cvjena/JeFaPaTo) the extraction of the EAR time series
- [CurlyBrace](https://github.com/iruletheworld/matplotlib-curly-brace) for the nice visualizations of curly braces :)

## Contact

For any queries, requests, or problems, please reach out to

- [Yuxuan Xie](yuxuan.xie@uni-jena.de)
- [Tim Büchner](tim.buechner@uni-jena.de)
