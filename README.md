# ReactRanker

[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)

A learning to rank (LTR) package for ranking chemical reactions. The reactions are encoded by [D-MPNN](https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00237) layers, and then passing the LTR layers. This package is developed basd on [Chemprop](https://github.com/cgrambow/chemprop/tree/reaction). The ```data``` class and ```train``` module were rewritten.

### Requirments
* rdkit
* pytorch
* torchvision
* numpy
* pandas
* scikit-learn
* scipy
* tensorflow
* tensorboardX
* tqdm

### Supportted method:
#### PointWise:
Regression  
Regression with gaussian piror distribution

Using the key words ```regression``` for regression or ```gauss_regression``` for regression with gaussian piror distribution.
#### PairWise:
RankNet  
BetaNet

Running the  ```regression``` for regression or ```gauss_regression``` for regression with gaussian piror distribution.
#### ListWise:
ListNet(@1)  
ListMLE
Uncertainty calibrated listwise

Note: This package is developing, the first release (v1) will happen within few months
