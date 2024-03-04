# EBPM - Eye Blinking Prototype Matching
This work presents a novel method for detecting eye blinking by establishing *eye state prototypes* to match blink patterns within eye aspect ratio (EAR) time series. In contrast to traditional methods, which mainly focus on the binary ON/OFF of blinkings, our method takes care of important diagnostic details such as blink speed, duration as well as inter-eye synchronicity. 

In an unsupervised manner, we learned prototypes from the existing blink patterns and established manually defined prototypes. Our research shows that both *unsupervised learned* and *manually defined prototypes* can reliably detect blink intervals and have comparable results, which offers potential diagnostic tools for identifying muscular or neural disorders. 

Under the principle of the "minimal working prototype", our aim is to establish the eye blink prototype with a minimum amount of work, enabling medical professionals without computer expertise to easily create their own prototypes to match specific patterns. This repository presents the source code of our approach and provides a demonstration in sample experiments. 

## Get Started
1. clone the repository
```bash
git clone
```
1. `cd` to EBPM
2. setup
```bash
conda create -n ebpm python=3.10 -y
conda activate ebpm
# conda install cudatoolkit -y
pip install jupyter
pip install -e .
```

## Usage
- Data
- Prototypes
- Experiments

## Citation

## License
Licensed under [MIT](Licensed.txt).
## Aknowledgements
TODO: stumpy, mediapipe, jefapato, curly brace...more?
## Contact
For any queries, please reach out to Yuxuan Xie at [yuxuan.xie@uni-jena.de](yuxuan.xie@uni-jena.de) or Tim Büchner at [tim.buechner@uni-jena.de](tim.buechner@uni-jena.de).