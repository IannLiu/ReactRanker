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

Using the key words ```regression``` to train regression model or ```gauss_regression``` to train regression model with gaussian piror distribution.
#### PairWise:
RankNet  

Running the  ```main_ranknet``` script to train ranknet model
#### ListWise:
ListNet(@1)  
ListMLE
UC-Listwise
Using the key words ```mle``` to train the ListMLE model, ```listnet``` to train the ListNet model, evidential_ranking to train UC-Listwise model

Note: This package is developing, the first release (v1) will be published within few months.
