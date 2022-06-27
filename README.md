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
Regression with gaussian piror distribution(Data and model uncertainty)  
#### PairWise:
RankNet  
BetaNet  
#### ListWise:
ListNet(@1)  
ListMLE  
#### Other Method:
##### Rank-Regression:
ListMLE-Regression  
ListMLE-Regression with gaussian distribution  
#### Rank-Classification:
ListMLE-Dirichlet distribution  

### Note:
#### Please cite this package if the following methods are used in your research:  
ListMLE-Regression  
ListMLE-Regression with gaussian distribution  
ListMLE-Dirichlet distribution  
BetaNet  
