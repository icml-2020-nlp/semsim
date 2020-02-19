
This folder provides the non-anonymized version of the CNN / Daily Mail summarization dataset for BART and the code to produce the dataset.
The dataset and codes are modified from [here](https://github.com/abisee/cnn-dailymail).

**Python 3 version**: This code is in Python 2. If you want a Python 3 version, see [@artmatsak's fork](https://github.com/artmatsak/cnn-dailymail). However, this will not produce the exact dataset we used for the experiment. 
<br>We recommend downloading the dataset from Option 1.

# Option 1: download the processed data

Download `cnn_dm.tar.gz`(493MB) from [here](https://drive.google.com/open?id=1goxgX-0_2Jo7cNAFrsb9BJTsrH7re6by)

|     Name     |                  SHA1SUM                 |
|:------------:|:----------------------------------------:|
| test.source  | 8f20fcbaa1f1705d418470ffa34abc2a940d3644 |
| test.target  | b2283a8bd0378a0e8b18aba0b925c12f25e73d56 |
| train.source | 91054e4258e5d2bf82414ae39da9ab43d379afdc |
| train.target | 35e808c302a31ece0c4b962aa8e174b8f48ac2c2 |
| val.source   | 5583ca6c9c81da2904c4923c0a50fa2f554e588f |
| val.target   | c8812748baf8dcbbd71af1479efb0cf9afcfd3ba |

# Option 2: process the data yourself

## 1. Replace make_datafiles.py
Clone git repository by [See](https://github.com/abisee/cnn-dailymail).
Replace `make_datafiles.py` with `make_datafiles.py` from this folder.

```
git clone https://github.com/abisee/cnn-dailymail.git
mv make_datafiles.py cnn-dailymail/make_datafiles.py
cd cnn-dailymail
```

## 2. Download data
Download and unzip the `stories` directories from [here](http://cs.nyu.edu/~kcho/DMQA/) for both CNN and Daily Mail. 

**Warning:** These files contain a few (114, in a dataset of over 300,000) examples for which the article text is missing - see for example `cnn/stories/72aba2f58178f2d19d3fae89d5f3e9a4686bc4bb.story`. The [Tensorflow code](https://github.com/abisee/pointer-generator) has been updated to discard these examples.

## 3. Process .source and .target files
Execute the following command.
```python make_datafiles.py /path/to/cnn/stories /path/to/dailymail/stories```

Replace `/path/to/cnn/stories` with the path to where you saved the `cnn/stories` directory that you downloaded; similarly for `dailymail/stories`.



