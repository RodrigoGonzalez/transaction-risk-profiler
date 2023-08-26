# The data
Again, to reiterate, this data is **very confidential**. Please do not put it on your personal machine or anywhere besides the mac minis!

The data is in `data/train_new.tar.gz`. You will cross validate within this dataset to find the best model.

This is event data and you need to determine if it's fraud. We start by taking a look at the data (use `pd.read_json`). There are 55 columns! What do these all mean? Many of them won't be useful for you, so a big part of the task is feature extraction.

You can use the `acct_type` to get the label. There are a few different possibilities for this value, but all we care about is fraud or not fraud.

Many of these columns may be useful to you. You can use Beautiful Soup and NLP techniques to get features out of the description column.


# Build your model
Build a really dumb model first as your baseline. Then keep trying different techniques and compare it against your model.


# Saving your model
After some exploratory analysis, you'll land on a model you're happy with. It's okay if some of your exploratory analysis isn't pretty code, but your model building code should be well written!

Now we'd like to build the model once and save it so we only have to do the time-consuming part of building the model once. We can save python objects with the `cPickle` module.

Here's how you can do some pickling:

```python
import random
import cPickle as pickle

class MyModel():
    def fit():
        pass
    def predict():
        return random.choice([True, False])

def get_data(datafile):
    ....
    return X, y

if __name__ == '__main__':
    X, y = get_data('data/train.json')
    model = MyModel()
    model.fit(X, y)
    with open('model.pkl', 'w') as f:
        pickle.dump(model, f)
```

Now I can reload the model from the pickle and use it to predict! No need to retrain :)

```python
with open('model.pkl') as f:
    model = pickle.load(f)

model.predict(...)
```

(14337, 44)
